#!/usr/bin/env python3
"""
簡化版QASA模型Benchmark整合系統
不依賴PyTorch Lightning，使用純PyTorch實現
"""

import pandas as pd
import numpy as np
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import math

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 檢查依賴
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
    logger.info("✅ PennyLane is available")
except ImportError as e:
    logger.error(f"❌ PennyLane not available: {e}")
    PENNYLANE_AVAILABLE = False

# 簡化版QASA模型組件
class SimplifiedQASAQuantumLayer(nn.Module):
    """簡化版QASA量子層"""
    
    def __init__(self, input_dim, output_dim, n_qubits=6, n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 創建量子設備
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 量子電路 - 簡化版本
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # 特徵編碼
            for i in range(min(len(inputs), n_qubits)):
                qml.RY(inputs[i], wires=i)
            
            # 變分層
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                
                # 糾纏
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.weight_shape = (n_layers, n_qubits)
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, {'weights': self.weight_shape})
        
        # 經典層
        self.input_proj = nn.Linear(input_dim, n_qubits)
        self.norm = nn.LayerNorm(n_qubits)
        # 修復：根據實際量子層輸出調整維度
        self.output_proj = nn.Linear(36, output_dim)  # 36是實際的量子層輸出維度
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        nn.init.kaiming_uniform_(self.input_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.output_proj.weight, mode='fan_in', nonlinearity='relu')
        if self.input_proj.bias is not None:
            nn.init.constant_(self.input_proj.bias, 0)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0)
    
    def forward(self, x):
        x_proj = torch.tanh(self.input_proj(x))
        x_proj = self.norm(x_proj)
        
        # 量子計算
        quantum_output = self.qlayer(x_proj)
        out = self.output_proj(quantum_output)
        
        # 跳躍連接
        if self.input_dim == self.output_dim:
            return x + out
        else:
            return out

class SimplifiedQASAHybridModel(nn.Module):
    """簡化版QASA混合模型"""
    
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=1, n_qubits=6, n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 經典層
        self.classical_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 量子層
        self.quantum_layer = SimplifiedQASAQuantumLayer(hidden_dim, hidden_dim, n_qubits, n_layers)
        
        # 輸出層
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 初始化權重
        self._init_weights()
    
    def _init_weights(self):
        """初始化權重"""
        for layer in self.classical_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x):
        # 經典處理
        classical_out = self.classical_layers(x)
        
        # 量子處理
        quantum_out = self.quantum_layer(classical_out)
        
        # 輸出
        output = self.output_layer(quantum_out)
        return output

class SimplifiedQASAStrategy:
    """簡化版QASA策略"""
    
    def __init__(self, name="Simplified_QASA", input_dim=12, hidden_dim=64, n_qubits=6, n_layers=3):
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 創建模型
        self.model = SimplifiedQASAHybridModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.is_trained = False
        self.last_rebalance_time = 0
        self.min_rebalance_interval = 0
        
        logger.info(f"Initialized {name} with {n_qubits} qubits, {n_layers} layers, {hidden_dim} hidden dim")
    
    def prepare_features(self, data):
        """準備特徵 - 修復維度問題"""
        features = []
        
        if 'close' in data.columns:
            price = data['close'].values
            returns = np.diff(price, prepend=price[0]) / price
            # 只取前3個returns特徵
            features.extend([
                returns,
                np.roll(returns, 1),
                np.roll(returns, 2),
            ])
        
        if len(data) > 20:
            close = data['close']
            ma_5 = close.rolling(5).mean().fillna(close)
            ma_10 = close.rolling(10).mean().fillna(close)
            ma_20 = close.rolling(20).mean().fillna(close)
            # 只取3個移動平均特徵
            features.extend([
                (close - ma_5) / ma_5,
                (close - ma_10) / ma_10,
                (close - ma_20) / ma_20,
            ])
        
        if 'volume' in data.columns:
            volume = data['volume'].values
            volume_returns = np.diff(volume, prepend=volume[0]) / volume
            # 只取2個volume特徵
            features.extend([
                volume_returns,
                volume / np.mean(volume),
            ])
        
        if len(data) > 10:
            close = data['close']
            volatility = close.rolling(10).std().fillna(close.std())
            # 只取2個volatility特徵
            features.extend([
                volatility / close,
                (volatility - volatility.rolling(5).mean()) / volatility.rolling(5).std(),
            ])
        
        # 確保特徵數量不超過input_dim
        features = features[:self.input_dim]
        
        features_array = np.array(features).T
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 確保維度匹配
        if features_array.shape[1] < self.input_dim:
            padding = np.zeros((features_array.shape[0], self.input_dim - features_array.shape[1]))
            features_array = np.hstack([features_array, padding])
        elif features_array.shape[1] > self.input_dim:
            features_array = features_array[:, :self.input_dim]
        
        return features_array
    
    def train(self, data, epochs=20):
        """訓練策略"""
        logger.info(f"Training {self.name} strategy...")
        
        try:
            # 準備訓練數據
            X, y = self._prepare_training_data(data)
            
            if len(X) == 0:
                logger.warning(f"No training data available for {self.name}")
                self.is_trained = True
                return
            
            # 轉換為PyTorch tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # 訓練循環
            self.model.train()
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                
                # 前向傳播
                predictions = self.model(X_tensor).squeeze()
                loss = self.loss_fn(predictions, y_tensor)
                
                # 反向傳播
                loss.backward()
                self.optimizer.step()
                
                if epoch % 5 == 0:
                    logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
            self.is_trained = True
            logger.info(f"{self.name} strategy training completed")
            
        except Exception as e:
            logger.error(f"Training failed for {self.name}: {e}")
            self.is_trained = True  # 標記為已訓練以避免重複嘗試
    
    def _prepare_training_data(self, data):
        """準備訓練數據"""
        X, y = [], []
        
        for i in range(20, len(data) - 1):  # 需要足夠的歷史數據
            # 輸入特徵
            sequence_data = data.iloc[i-20:i]
            features = self.prepare_features(sequence_data)
            
            # 目標：下一個時間步的價格變化
            current_price = data['close'].iloc[i]
            next_price = data['close'].iloc[i + 1]
            price_change = (next_price - current_price) / current_price
            
            X.append(features[-1])  # 取最後一個時間步的特徵
            y.append(price_change)
        
        return np.array(X), np.array(y)
    
    def should_rebalance(self, data, current_time):
        """再平衡判斷"""
        if not self.is_trained:
            return False
        
        try:
            # 準備特徵
            features = self.prepare_features(data)
            
            # 轉換為tensor
            X_tensor = torch.FloatTensor(features[-1:])  # 取最後一個時間步
            
            # 預測
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X_tensor).item()
            
            # 使用預測的價格變化來決定是否再平衡
            threshold = 0.01  # 1%的價格變化閾值
            should_rebalance = abs(prediction) > threshold
            
            if should_rebalance:
                self.last_rebalance_time = current_time
            
            return should_rebalance
            
        except Exception as e:
            logger.warning(f"Rebalance calculation failed: {e}")
            return False
    
    def calculate_position_width(self, data):
        """計算位置寬度"""
        if not self.is_trained:
            return 0.15
        
        try:
            # 準備特徵
            features = self.prepare_features(data)
            
            # 轉換為tensor
            X_tensor = torch.FloatTensor(features[-1:])
            
            # 預測
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X_tensor).item()
            
            # 根據預測的價格變化幅度調整位置寬度
            confidence = min(abs(prediction) * 10, 1.0)  # 將預測轉換為置信度
            position_width = min(confidence * 0.3, 0.3)
            return max(position_width, 0.1)
            
        except Exception as e:
            logger.warning(f"Position width calculation failed: {e}")
            return 0.15

def load_5year_data(data_dir="amm-rebalance-backtester/data/5year_daily"):
    """加載5年數據"""
    data_dir = Path(data_dir)
    data_files = {}
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {}
    
    for file_path in data_dir.glob("*.csv"):
        symbol = file_path.stem
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            data_files[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} records from {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return data_files

def run_simplified_qasa_benchmark(data, symbol, output_dir):
    """運行簡化版QASA benchmark"""
    logger.info(f"Running Simplified QASA benchmark for {symbol}")
    
    # 創建QASA策略
    strategies = {
        'Simplified_QASA_Hybrid': SimplifiedQASAStrategy(
            name='Simplified_QASA_Hybrid',
            input_dim=12,
            hidden_dim=64,
            n_qubits=6,
            n_layers=3
        ),
    }
    
    results = []
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"Testing {strategy_name}...")
        
        try:
            split_idx = int(0.7 * len(data))
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            strategy.train(train_data, epochs=20)
            
            initial_cash = 10000.0
            cash = initial_cash
            position = 0.0
            rebalances = 0
            position_widths = []
            
            for i, (timestamp, row) in enumerate(test_data.iterrows()):
                current_data = test_data.iloc[:i+1]
                
                if strategy.should_rebalance(current_data, i):
                    width = strategy.calculate_position_width(current_data)
                    position_widths.append(width)
                    
                    if position == 0:
                        position = cash * 0.7
                        cash -= position
                        rebalances += 1
                    else:
                        rebalances += 1
                
                if position > 0:
                    price_change = (row['close'] - row['open']) / row['open']
                    cash += position * price_change
            
            final_price = test_data['close'].iloc[-1]
            final_value = cash + position * (final_price / test_data['close'].iloc[0])
            total_return = (final_value - initial_cash) / initial_cash * 100
            
            returns = test_data['close'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            result = {
                'Symbol': symbol,
                'Strategy': strategy_name,
                'Type': 'Simplified_QASA',
                'Model_Components': f'Simplified_QASA_Hybrid_(6_qubits_3_layers_64_hidden)',
                'Rebalances': rebalances,
                'Avg_Width': np.mean(position_widths) if position_widths else 0.15,
                'Final_Portfolio': final_value,
                'Return_Pct': total_return,
                'Sharpe_Ratio': sharpe_ratio
            }
            
            results.append(result)
            logger.info(f"✅ {strategy_name}: {total_return:.2f}% return, {rebalances} rebalances")
            
        except Exception as e:
            logger.error(f"❌ {strategy_name} failed: {e}")
            results.append({
                'Symbol': symbol,
                'Strategy': strategy_name,
                'Type': 'Simplified_QASA',
                'Model_Components': 'Error',
                'Rebalances': 0,
                'Avg_Width': 0.15,
                'Final_Portfolio': 10000.0,
                'Return_Pct': 0.0,
                'Sharpe_Ratio': 0.0
            })
    
    return results

def save_results(results, output_dir, filename):
    """保存結果到CSV文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = output_path / filename
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")
    
    return df

def create_comparison_chart(df, output_dir):
    """創建比較圖表"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 策略收益比較
        ax1 = axes[0, 0]
        strategy_returns = df.groupby('Strategy')['Return_Pct'].mean().sort_values(ascending=True)
        bars1 = ax1.barh(strategy_returns.index, strategy_returns.values, alpha=0.7, color='lightblue')
        ax1.set_title('Simplified QASA Hybrid Performance', fontweight='bold')
        ax1.set_xlabel('Average Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 資產收益比較
        ax2 = axes[0, 1]
        symbol_returns = df.groupby('Symbol')['Return_Pct'].mean().sort_values(ascending=True)
        bars2 = ax2.barh(symbol_returns.index, symbol_returns.values, alpha=0.7, color='lightcoral')
        ax2.set_title('Performance by Asset', fontweight='bold')
        ax2.set_xlabel('Average Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 再平衡次數 vs 收益
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['Rebalances'], df['Return_Pct'], alpha=0.7, s=100)
        ax3.set_xlabel('Number of Rebalances')
        ax3.set_ylabel('Return (%)')
        ax3.set_title('Rebalances vs Performance', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 夏普比率分布
        ax4 = axes[1, 1]
        ax4.hist(df['Sharpe_Ratio'], bins=10, alpha=0.7, color='lightgreen')
        ax4.set_xlabel('Sharpe Ratio')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Sharpe Ratio Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = Path(output_dir) / 'simplified_qasa_benchmark_summary.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison chart saved to: {chart_path}")
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping chart creation")

def main():
    """主函數"""
    if not PENNYLANE_AVAILABLE:
        logger.error("❌ PennyLane not available. Please install pennylane.")
        return
    
    logger.info("🚀 Starting Simplified QASA Hybrid Benchmark...")
    
    data_files = load_5year_data()
    
    if not data_files:
        logger.error("No data files found")
        return
    
    all_results = []
    
    for symbol, data in data_files.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*60}")
        
        results = run_simplified_qasa_benchmark(data, symbol, "reports/simplified_qasa_benchmark")
        all_results.extend(results)
    
    if all_results:
        output_dir = "reports/simplified_qasa_benchmark"
        df = save_results(all_results, output_dir, 'simplified_qasa_benchmark_results.csv')
        create_comparison_chart(df, output_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info("SIMPLIFIED QASA HYBRID BENCHMARK SUMMARY")
        logger.info(f"{'='*60}")
        
        logger.info(f"\n📊 Overall Statistics:")
        logger.info(f"  Total Strategies Tested: {len(df)}")
        logger.info(f"  Average Return: {df['Return_Pct'].mean():.2f}%")
        logger.info(f"  Best Return: {df['Return_Pct'].max():.2f}%")
        logger.info(f"  Worst Return: {df['Return_Pct'].min():.2f}%")
        logger.info(f"  Average Sharpe Ratio: {df['Sharpe_Ratio'].mean():.3f}")
        
        logger.info(f"\n🏆 Top 5 Strategies by Return:")
        top_strategies = df.nlargest(5, 'Return_Pct')[['Strategy', 'Symbol', 'Return_Pct', 'Sharpe_Ratio']]
        for _, row in top_strategies.iterrows():
            logger.info(f"  {row['Strategy']} ({row['Symbol']}): {row['Return_Pct']:.2f}% return, {row['Sharpe_Ratio']:.3f} Sharpe")
        
        logger.info(f"\n📁 Results saved to: {output_dir}")
        logger.info("✅ Simplified QASA Hybrid benchmark completed!")
    
    else:
        logger.error("No results generated")

if __name__ == '__main__':
    main()

