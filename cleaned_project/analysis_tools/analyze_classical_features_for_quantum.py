#!/usr/bin/env python3
"""
分析經典機器學習的特徵重要性，並設計量子機器學習學習這些重要特徵的方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ClassicalFeatureAnalyzer:
    """經典機器學習特徵重要性分析器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importance_results = {}
        
    def create_comprehensive_features(self, price_data):
        """創建全面的特徵集"""
        df = price_data.copy()
        
        # 1. 基礎價格特徵
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_ma_ratio'] = df['close'] / df['close'].rolling(20).mean()
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 2. 移動平均特徵
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            df[f'price_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
        
        # 3. 技術指標
        df['rsi'] = self._calculate_rsi(df['close'])
        macd_line, signal_line, histogram = self._calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        df['atr'] = self._calculate_atr(df)
        
        # 4. 波動性特徵
        for period in [5, 10, 20, 30]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_{period}_annualized'] = df[f'volatility_{period}'] * np.sqrt(252)
        
        df['volatility_ewma'] = df['returns'].ewm(span=20).std()
        df['vol_of_vol'] = df['volatility_20'].rolling(10).std()
        df['vol_percentile'] = df['volatility_20'].rolling(100).rank(pct=True)
        df['vol_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(50).quantile(0.8)).astype(int)
        
        # 5. 成交量特徵
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        df['volume_change'] = df['volume'].pct_change()
        df['volume_volatility'] = df['volume_change'].rolling(20).std()
        
        # 6. 時間特徵
        if hasattr(df.index, 'hour'):
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
        else:
            # 如果沒有時間索引，創建模擬時間特徵
            df['hour_sin'] = np.sin(2 * np.pi * np.arange(len(df)) / 24)
            df['hour_cos'] = np.cos(2 * np.pi * np.arange(len(df)) / 24)
            df['day_of_week'] = np.arange(len(df)) % 7
            df['day_of_month'] = np.arange(len(df)) % 30
        
        # 7. 滯後特徵
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # 8. 交互特徵
        df['vol_volume_interaction'] = df['volatility_20'] * df['volume_ratio_20']
        df['momentum_rsi_interaction'] = df['returns'] * (df['rsi'] - 50) / 50
        df['vol_regime_interaction'] = df['volatility_20'] * df['vol_regime']
        
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """計算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """計算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """計算布林帶"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    def _calculate_atr(self, df, window=14):
        """計算ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window).mean()
        return atr
    
    def create_targets(self, df, rebalance_threshold=0.01):
        """創建目標變量"""
        # 再平衡決策 - 使用更低的閾值確保有足夠的正樣本
        price_deviation = abs(df['close'] / df['close'].rolling(20).mean() - 1)
        df['should_rebalance'] = (price_deviation > rebalance_threshold).astype(int)
        
        # 確保有足夠的正樣本
        if df['should_rebalance'].sum() < 10:
            # 如果正樣本太少，降低閾值
            df['should_rebalance'] = (price_deviation > rebalance_threshold * 0.5).astype(int)
        
        # 未來波動性
        df['future_volatility'] = df['returns'].rolling(20).std().shift(-20)
        
        return df
    
    def analyze_feature_importance(self, price_data, model_types=['random_forest', 'gradient_boosting']):
        """分析特徵重要性"""
        logger.info("Creating comprehensive features...")
        
        # 創建特徵
        df = self.create_comprehensive_features(price_data)
        df = self.create_targets(df)
        
        # 移除缺失值
        df = df.dropna()
        
        # 分離特徵和目標
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                     'should_rebalance', 'future_volatility']]
        
        X = df[feature_cols]
        y_rebalance = df['should_rebalance']
        y_volatility = df['future_volatility']
        
        logger.info(f"Created {len(feature_cols)} features for {len(X)} samples")
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_rebalance, test_size=0.2, random_state=42, stratify=y_rebalance
        )
        
        # 標準化特徵
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model...")
            
            # 訓練模型
            if model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                continue
            
            model.fit(X_train_scaled, y_train)
            
            # 獲取特徵重要性
            feature_importance = pd.Series(
                model.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False)
            
            # 計算性能指標
            accuracy = model.score(X_test_scaled, y_test)
            
            results[model_type] = {
                'model': model,
                'feature_importance': feature_importance,
                'accuracy': accuracy,
                'top_features': feature_importance.head(20).to_dict()
            }
            
            logger.info(f"{model_type} accuracy: {accuracy:.4f}")
            logger.info(f"Top 10 features: {list(feature_importance.head(10).index)}")
        
        self.feature_importance_results = results
        return results
    
    def plot_feature_importance(self, save_path="reports/classical_feature_analysis"):
        """繪製特徵重要性圖表"""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        for i, (model_type, result) in enumerate(self.feature_importance_results.items()):
            ax = axes[i//2, i%2]
            
            # 取前15個最重要的特徵
            top_features = result['feature_importance'].head(15)
            
            ax.barh(range(len(top_features)), top_features.values)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features.index, fontsize=8)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{model_type.title()} - Top 15 Features\nAccuracy: {result["accuracy"]:.4f}')
            ax.invert_yaxis()
        
        # 合併特徵重要性
        ax = axes[1, 1]
        all_features = set()
        for result in self.feature_importance_results.values():
            all_features.update(result['feature_importance'].head(10).index)
        
        combined_importance = {}
        for feature in all_features:
            importance_sum = 0
            count = 0
            for result in self.feature_importance_results.values():
                if feature in result['feature_importance'].index:
                    importance_sum += result['feature_importance'][feature]
                    count += 1
            combined_importance[feature] = importance_sum / count if count > 0 else 0
        
        combined_series = pd.Series(combined_importance).sort_values(ascending=False).head(15)
        ax.barh(range(len(combined_series)), combined_series.values)
        ax.set_yticks(range(len(combined_series)))
        ax.set_yticklabels(combined_series.index, fontsize=8)
        ax.set_xlabel('Average Feature Importance')
        ax.set_title('Combined Top 15 Features\n(Average across models)')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_importance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plots saved to {save_path}")

class QuantumFeatureDesigner:
    """量子特徵設計器 - 基於經典ML重要特徵設計量子特徵"""
    
    def __init__(self, top_classical_features):
        self.top_classical_features = top_classical_features
        self.quantum_feature_mapping = {}
        
    def design_quantum_features(self, n_qubits=6):
        """設計量子特徵映射"""
        logger.info(f"Designing quantum features for {n_qubits} qubits...")
        
        # 特徵分類
        price_features = [f for f in self.top_classical_features if 'returns' in f or 'price' in f or 'sma' in f or 'ema' in f]
        volatility_features = [f for f in self.top_classical_features if 'volatility' in f or 'vol_' in f]
        volume_features = [f for f in self.top_classical_features if 'volume' in f]
        technical_features = [f for f in self.top_classical_features if any(x in f for x in ['rsi', 'macd', 'bb_', 'atr'])]
        interaction_features = [f for f in self.top_classical_features if 'interaction' in f]
        
        # 量子特徵映射策略
        quantum_features = []
        
        # 1. 價格特徵 (2個量子比特)
        if len(price_features) >= 2:
            quantum_features.extend([
                {
                    'name': 'price_momentum',
                    'classical_features': price_features[:2],
                    'quantum_encoding': 'angle_encoding',
                    'qubit_index': 0,
                    'description': 'Price momentum and trend'
                },
                {
                    'name': 'price_ma_ratio',
                    'classical_features': [f for f in price_features if 'ratio' in f][:1],
                    'quantum_encoding': 'angle_encoding',
                    'qubit_index': 1,
                    'description': 'Price to moving average ratio'
                }
            ])
        
        # 2. 波動性特徵 (2個量子比特)
        if len(volatility_features) >= 2:
            quantum_features.extend([
                {
                    'name': 'volatility_level',
                    'classical_features': volatility_features[:1],
                    'quantum_encoding': 'angle_encoding',
                    'qubit_index': 2,
                    'description': 'Current volatility level'
                },
                {
                    'name': 'volatility_regime',
                    'classical_features': [f for f in volatility_features if 'regime' in f or 'percentile' in f][:1],
                    'quantum_encoding': 'angle_encoding',
                    'qubit_index': 3,
                    'description': 'Volatility regime state'
                }
            ])
        
        # 3. 技術指標特徵 (1個量子比特)
        if len(technical_features) >= 1:
            quantum_features.append({
                'name': 'technical_signal',
                'classical_features': technical_features[:1],
                'quantum_encoding': 'angle_encoding',
                'qubit_index': 4,
                'description': 'Technical indicator signal'
            })
        
        # 4. 成交量特徵 (1個量子比特)
        if len(volume_features) >= 1:
            quantum_features.append({
                'name': 'volume_signal',
                'classical_features': volume_features[:1],
                'quantum_encoding': 'angle_encoding',
                'qubit_index': 5,
                'description': 'Volume signal'
            })
        
        self.quantum_feature_mapping = {
            'n_qubits': n_qubits,
            'features': quantum_features,
            'feature_categories': {
                'price': price_features,
                'volatility': volatility_features,
                'volume': volume_features,
                'technical': technical_features,
                'interaction': interaction_features
            }
        }
        
        return self.quantum_feature_mapping
    
    def create_quantum_encoding_function(self):
        """創建量子特徵編碼函數"""
        encoding_code = '''
def encode_quantum_features(classical_features, quantum_mapping):
    """
    將經典特徵編碼為量子特徵
    
    Args:
        classical_features: 經典特徵數據框
        quantum_mapping: 量子特徵映射配置
    
    Returns:
        quantum_features: 編碼後的量子特徵
    """
    n_qubits = quantum_mapping['n_qubits']
    quantum_features = np.zeros((len(classical_features), n_qubits))
    
    for feature_config in quantum_mapping['features']:
        qubit_idx = feature_config['qubit_index']
        classical_feature_names = feature_config['classical_features']
        
        # 計算特徵值
        feature_value = 0
        for feature_name in classical_feature_names:
            if feature_name in classical_features.columns:
                feature_value += classical_features[feature_name].values
        
        # 角度編碼 [0, 2π]
        if feature_config['quantum_encoding'] == 'angle_encoding':
            # 標準化到 [0, 1] 然後縮放到 [0, 2π]
            feature_value = np.clip(feature_value, -3, 3)  # 限制極值
            feature_value = (feature_value + 3) / 6  # 標準化到 [0, 1]
            quantum_features[:, qubit_idx] = feature_value * 2 * np.pi
    
    return quantum_features

def create_quantum_circuit(n_qubits=6, n_layers=3):
    """
    創建基於重要特徵的量子電路
    
    Args:
        n_qubits: 量子比特數
        n_layers: 變分層數
    
    Returns:
        quantum_circuit: 量子電路函數
    """
    import pennylane as qml
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev, interface='torch')
    def quantum_circuit(features, weights):
        # 特徵編碼層
        for i in range(n_qubits):
            qml.RY(features[i], wires=i)
            qml.RZ(features[i] * 0.5, wires=i)
        
        # 變分層
        for layer in range(n_layers):
            # 旋轉門
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
                qml.RZ(weights[layer, i + n_qubits], wires=i)
            
            # 糾纏層 - 基於特徵重要性設計
            # 價格和波動性特徵糾纏
            qml.CNOT(wires=[0, 2])  # price_momentum -> volatility_level
            qml.CNOT(wires=[1, 3])  # price_ma_ratio -> volatility_regime
            
            # 技術指標和成交量糾纏
            qml.CNOT(wires=[4, 5])  # technical_signal -> volume_signal
            
            # 全局糾纏
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # 測量期望值
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return quantum_circuit
'''
        
        return encoding_code
    
    def generate_quantum_implementation(self, save_path="reports/classical_feature_analysis"):
        """生成量子實現代碼"""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 生成量子特徵設計報告
        report_content = f"""
# 基於經典ML重要特徵的量子特徵設計

## 經典ML重要特徵分析

### 前10個最重要特徵:
{list(self.top_classical_features[:10])}

### 特徵分類:
- 價格特徵: {len([f for f in self.top_classical_features if 'returns' in f or 'price' in f or 'sma' in f or 'ema' in f])}個
- 波動性特徵: {len([f for f in self.top_classical_features if 'volatility' in f or 'vol_' in f])}個
- 成交量特徵: {len([f for f in self.top_classical_features if 'volume' in f])}個
- 技術指標特徵: {len([f for f in self.top_classical_features if any(x in f for x in ['rsi', 'macd', 'bb_', 'atr'])])}個
- 交互特徵: {len([f for f in self.top_classical_features if 'interaction' in f])}個

## 量子特徵映射設計

### 量子比特分配:
{self.quantum_feature_mapping}

### 量子電路設計原則:
1. 基於經典ML特徵重要性進行量子比特分配
2. 使用角度編碼將經典特徵映射到量子態
3. 設計糾纏模式以捕捉特徵間的相關性
4. 優化變分層以學習重要的特徵組合

### 實現建議:
1. 使用前6個最重要的經典特徵作為量子特徵基礎
2. 採用角度編碼方式將特徵值映射到[0, 2π]範圍
3. 設計特定的糾纏模式以捕捉價格-波動性關係
4. 使用多層變分電路學習複雜的特徵交互
"""
        
        with open(f"{save_path}/quantum_feature_design_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 生成量子實現代碼
        quantum_code = self.create_quantum_encoding_function()
        with open(f"{save_path}/quantum_feature_implementation.py", 'w', encoding='utf-8') as f:
            f.write(quantum_code)
        
        logger.info(f"Quantum feature design saved to {save_path}")

def main():
    """主函數"""
    logger.info("🚀 Starting classical feature analysis for quantum learning...")
    
    # 創建示例數據
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # 模擬價格數據
    price_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.random.rand(1000) * 2,
        'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.random.rand(1000) * 2,
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    price_data['close'] = price_data['open'] + np.random.randn(1000) * 0.5
    price_data['high'] = np.maximum(price_data['open'], price_data['close']) + np.random.rand(1000)
    price_data['low'] = np.minimum(price_data['open'], price_data['close']) - np.random.rand(1000)
    
    # 分析經典特徵重要性
    analyzer = ClassicalFeatureAnalyzer()
    results = analyzer.analyze_feature_importance(price_data)
    
    # 繪製特徵重要性圖表
    analyzer.plot_feature_importance()
    
    # 獲取最重要的特徵
    all_features = set()
    for result in results.values():
        all_features.update(result['feature_importance'].head(10).index)
    
    # 計算平均重要性
    combined_importance = {}
    for feature in all_features:
        importance_sum = 0
        count = 0
        for result in results.values():
            if feature in result['feature_importance'].index:
                importance_sum += result['feature_importance'][feature]
                count += 1
        combined_importance[feature] = importance_sum / count if count > 0 else 0
    
    top_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
    top_feature_names = [f[0] for f in top_features]
    
    logger.info(f"Top 10 most important features: {top_feature_names[:10]}")
    
    # 設計量子特徵
    quantum_designer = QuantumFeatureDesigner(top_feature_names)
    quantum_mapping = quantum_designer.design_quantum_features(n_qubits=6)
    
    # 生成量子實現
    quantum_designer.generate_quantum_implementation()
    
    logger.info("✅ Classical feature analysis for quantum learning completed!")
    logger.info("📁 Results saved to reports/classical_feature_analysis/")

if __name__ == "__main__":
    main()
