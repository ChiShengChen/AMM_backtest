"""
改進的量子機器學習訓練與測試演示腳本
展示量子模型的最佳實踐訓練和測試流程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 量子計算庫
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, TwoLocal
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.primitives import Sampler
    import pennylane as qml
    QUANTUM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Quantum computing libraries available")
except ImportError as e:
    QUANTUM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Quantum computing libraries not available: {e}")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantumFeatureEngineer:
    """量子特徵工程器"""
    
    def __init__(self, n_features: int = 8):
        self.n_features = n_features
        self.scaler = MinMaxScaler(feature_range=(0, 2*np.pi))  # 量子特徵需要0到2π範圍
        self.feature_names = []
        
    def create_quantum_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """創建適合量子模型的特徵"""
        logger.info("Creating quantum-compatible features...")
        
        features = pd.DataFrame(index=price_data.index)
        
        # 1. 基本價格特徵
        features['returns'] = price_data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['price_momentum'] = price_data['close'] / price_data['close'].shift(5) - 1
        features['volume_ratio'] = price_data['volume'] / price_data['volume'].rolling(20).mean()
        
        # 2. 技術指標
        features['rsi'] = self._calculate_rsi(price_data['close'])
        features['macd'] = self._calculate_macd(price_data['close'])
        features['bollinger_position'] = self._calculate_bollinger_position(price_data['close'])
        
        # 3. 時間特徵
        features['hour_sin'] = np.sin(2 * np.pi * price_data.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * price_data.index.hour / 24)
        
        # 4. 選擇最重要的特徵
        features = features.dropna()
        
        # 選擇前n_features個特徵
        if len(features.columns) > self.n_features:
            # 簡單的特徵選擇：選擇方差最大的特徵
            feature_vars = features.var().sort_values(ascending=False)
            selected_features = feature_vars.head(self.n_features).index.tolist()
            features = features[selected_features]
        
        # 填充缺失值
        features = features.fillna(features.mean())
        
        # 標準化到量子範圍
        features_scaled = self.scaler.fit_transform(features)
        
        self.feature_names = features.columns.tolist()
        logger.info(f"Created {len(self.feature_names)} quantum features: {self.feature_names}")
        
        return features_scaled
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """計算RSI指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # 歸一化到0-1
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """計算MACD指標"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd / prices  # 歸一化
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.Series:
        """計算布林帶位置"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        position = (prices - lower) / (upper - lower)
        return position.fillna(0.5)  # 填充中間值

class QiskitVQCTrainer:
    """VQE Classifier訓練器"""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, feature_dim: int = 8):
        if not QUANTUM_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.vqc = None
        self.training_history = []
        
    def train(self, X: np.ndarray, y: np.ndarray, max_iter: int = 100) -> dict:
        """訓練VQC模型"""
        logger.info(f"Training VQE Classifier with {len(X)} samples, {X.shape[1]} features")
        
        # 創建特徵映射
        feature_map = ZZFeatureMap(feature_dimension=self.feature_dim, reps=1)
        
        # 創建變分形式
        ansatz = TwoLocal(self.feature_dim, ['ry', 'rz'], 'cz', reps=self.n_layers)
        
        # 創建VQC
        self.vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=max_iter),
            sampler=Sampler()
        )
        
        # 訓練模型
        start_time = datetime.now()
        self.vqc.fit(X, y)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 評估模型
        y_pred = self.vqc.predict(X)
        y_pred_proba = self.vqc.predict_proba(X)
        
        accuracy = accuracy_score(y, y_pred)
        
        logger.info(f"VQC training completed in {training_time:.2f}s, accuracy: {accuracy:.4f}")
        
        return {
            'model': self.vqc,
            'accuracy': accuracy,
            'training_time': training_time,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if self.vqc is None:
            raise ValueError("Model not trained")
        return self.vqc.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """預測概率"""
        if self.vqc is None:
            raise ValueError("Model not trained")
        return self.vqc.predict_proba(X)

class PennyLaneQNNTrainer:
    """QNN訓練器"""
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, feature_dim: int = 8):
        if not QUANTUM_AVAILABLE:
            raise ImportError("PennyLane not available")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_dim = feature_dim
        self.device = qml.device('default.qubit', wires=n_qubits)
        self.weights = None
        self.circuit = None
        self.training_history = []
        
    def _quantum_circuit(self, features, weights):
        """量子電路"""
        # 特徵編碼
        for i, feature in enumerate(features[:self.n_qubits]):
            qml.RY(feature, wires=i)
        
        # 變分層
        for layer in range(self.n_layers):
            # 旋轉門
            for i in range(self.n_qubits):
                qml.RY(weights[layer][i], wires=i)
                qml.RZ(weights[layer][i + self.n_qubits], wires=i)
            
            # 糾纏層
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # 測量期望值
        return qml.expval(qml.PauliZ(0))
    
    def train(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 50, learning_rate: float = 0.1) -> dict:
        """訓練QNN模型"""
        logger.info(f"Training QNN with {len(X)} samples, {X.shape[1]} features")
        
        # 初始化權重
        self.weights = np.random.uniform(0, 2*np.pi, (self.n_layers, 2*self.n_qubits))
        
        # 創建QNode
        self.circuit = qml.QNode(self._quantum_circuit, self.device)
        
        # 優化器
        opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
        
        # 訓練循環
        start_time = datetime.now()
        for epoch in range(n_epochs):
            cost = 0
            for i in range(len(X)):
                prediction = self.circuit(X[i], self.weights)
                cost += (prediction - y[i]) ** 2
            
            cost /= len(X)
            self.weights = opt.step(lambda w: cost, self.weights)
            
            self.training_history.append(cost)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Cost: {cost:.4f}")
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 評估模型
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        logger.info(f"QNN training completed in {training_time:.2f}s, accuracy: {accuracy:.4f}")
        
        return {
            'model': self,
            'accuracy': accuracy,
            'training_time': training_time,
            'training_history': self.training_history,
            'predictions': y_pred
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if self.weights is None:
            raise ValueError("Model not trained")
        
        predictions = []
        for i in range(len(X)):
            pred = self.circuit(X[i], self.weights)
            predictions.append(1 if pred > 0 else 0)
        
        return np.array(predictions)

class QuantumModelComparator:
    """量子模型比較器"""
    
    def __init__(self):
        self.results = {}
        
    def compare_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """比較不同量子模型"""
        logger.info("Comparing quantum models...")
        
        results = {}
        
        if QUANTUM_AVAILABLE:
            # 1. VQE Classifier
            try:
                logger.info("Training VQE Classifier...")
                vqc_trainer = QiskitVQCTrainer(n_qubits=4, n_layers=2, feature_dim=X_train.shape[1])
                vqc_result = vqc_trainer.train(X_train, y_train, max_iter=50)
                
                # 測試
                y_pred = vqc_trainer.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                results['Qiskit_VQC'] = {
                    'train_accuracy': vqc_result['accuracy'],
                    'test_accuracy': test_accuracy,
                    'training_time': vqc_result['training_time'],
                    'predictions': y_pred
                }
                
            except Exception as e:
                logger.error(f"VQE Classifier training failed: {e}")
                results['Qiskit_VQC'] = {'error': str(e)}
            
            # 2. QNN
            try:
                logger.info("Training QNN...")
                pennylane_trainer = PennyLaneQNNTrainer(n_qubits=4, n_layers=2, feature_dim=X_train.shape[1])
                pennylane_result = pennylane_trainer.train(X_train, y_train, n_epochs=30)
                
                # 測試
                y_pred = pennylane_trainer.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                results['PennyLane_QNN'] = {
                    'train_accuracy': pennylane_result['accuracy'],
                    'test_accuracy': test_accuracy,
                    'training_time': pennylane_result['training_time'],
                    'training_history': pennylane_result['training_history'],
                    'predictions': y_pred
                }
                
            except Exception as e:
                logger.error(f"QNN training failed: {e}")
                results['PennyLane_QNN'] = {'error': str(e)}
        
        # 3. 經典模型對比
        logger.info("Training classical Random Forest for comparison...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred_rf = rf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        
        results['Random_Forest'] = {
            'train_accuracy': accuracy_score(y_train, rf.predict(X_train)),
            'test_accuracy': rf_accuracy,
            'training_time': 0.1,  # 經典模型很快
            'predictions': y_pred_rf
        }
        
        self.results = results
        return results
    
    def create_comparison_report(self, save_path: str = "reports/quantum_training_evaluation"):
        """創建比較報告"""
        if not self.results:
            logger.warning("No results available for comparison")
            return
        
        # 創建目錄
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 1. 性能比較圖表
        self._plot_performance_comparison(save_path)
        
        # 2. 訓練時間比較
        self._plot_training_time_comparison(save_path)
        
        # 3. 預測分布比較
        self._plot_prediction_comparison(save_path)
        
        # 4. 保存結果
        self._save_results(save_path)
        
        logger.info(f"Quantum comparison report saved to {save_path}")
    
    def _plot_performance_comparison(self, save_path: str):
        """繪製性能比較圖表"""
        plt.figure(figsize=(15, 10))
        
        # 提取有效結果
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            logger.warning("No valid results to plot")
            return
        
        # 1. 準確率比較
        plt.subplot(2, 3, 1)
        models = list(valid_results.keys())
        train_accs = [valid_results[m]['train_accuracy'] for m in models]
        test_accs = [valid_results[m]['test_accuracy'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
        plt.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 訓練時間比較
        plt.subplot(2, 3, 2)
        training_times = [valid_results[m]['training_time'] for m in models]
        plt.bar(models, training_times, alpha=0.8, color='orange')
        plt.xlabel('Models')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. 過擬合分析
        plt.subplot(2, 3, 3)
        overfitting = [train_accs[i] - test_accs[i] for i in range(len(models))]
        plt.bar(models, overfitting, alpha=0.8, color='red')
        plt.xlabel('Models')
        plt.ylabel('Overfitting (Train - Test)')
        plt.title('Overfitting Analysis')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. 綜合評分
        plt.subplot(2, 3, 4)
        # 綜合評分：測試準確率 - 過擬合 - 訓練時間懲罰
        max_time = max(training_times)
        scores = []
        for i, model in enumerate(models):
            score = test_accs[i] - overfitting[i] * 0.5 - (training_times[i] / max_time) * 0.1
            scores.append(max(0, score))
        
        plt.bar(models, scores, alpha=0.8, color='green')
        plt.xlabel('Models')
        plt.ylabel('Composite Score')
        plt.title('Composite Performance Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. 預測分布（如果有PennyLane結果）
        if 'PennyLane_QNN' in valid_results and 'training_history' in valid_results['PennyLane_QNN']:
            plt.subplot(2, 3, 5)
            history = valid_results['PennyLane_QNN']['training_history']
            plt.plot(history)
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.title('QNN Training History')
            plt.grid(True, alpha=0.3)
        
        # 6. 模型穩定性（使用交叉驗證結果）
        plt.subplot(2, 3, 6)
        # 這裡可以添加穩定性分析
        plt.text(0.5, 0.5, 'Stability Analysis\n(To be implemented)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Model Stability')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/quantum_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_time_comparison(self, save_path: str):
        """繪製訓練時間比較"""
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            return
        
        plt.figure(figsize=(10, 6))
        models = list(valid_results.keys())
        times = [valid_results[m]['training_time'] for m in models]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        plt.bar(models, times, color=colors[:len(models)], alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel('Training Time (seconds)')
        plt.title('Quantum vs Classical Training Time Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for i, v in enumerate(times):
            plt.text(i, v + max(times)*0.01, f'{v:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_comparison(self, save_path: str):
        """繪製預測比較"""
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            return
        
        plt.figure(figsize=(12, 8))
        
        n_models = len(valid_results)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        if n_models == 1:
            axes = [axes]
        elif n_models == 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(valid_results.items()):
            if i >= len(axes):
                break
                
            predictions = result['predictions']
            
            # 預測分布
            axes[i].hist(predictions, bins=2, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{model_name} Predictions')
            axes[i].set_xlabel('Prediction')
            axes[i].set_ylabel('Count')
            axes[i].grid(True, alpha=0.3)
        
        # 隱藏多餘的子圖
        for i in range(len(valid_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/prediction_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, save_path: str):
        """保存結果到CSV"""
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            return
        
        # 創建結果DataFrame
        results_data = []
        for model_name, result in valid_results.items():
            results_data.append({
                'Model': model_name,
                'Train_Accuracy': result['train_accuracy'],
                'Test_Accuracy': result['test_accuracy'],
                'Training_Time': result['training_time'],
                'Overfitting': result['train_accuracy'] - result['test_accuracy']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f"{save_path}/quantum_model_results.csv", index=False)
        
        logger.info(f"Results saved to {save_path}/quantum_model_results.csv")

def create_sample_data(days: int = 500) -> pd.DataFrame:
    """創建樣本數據"""
    logger.info(f"Creating {days} days of sample data...")
    
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    # 模擬價格數據
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # 創建OHLCV數據
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        daily_vol = abs(np.random.normal(0, 0.01))
        high = price * (1 + daily_vol)
        low = price * (1 - daily_vol)
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Sample data created: {len(df)} records")
    return df

def main():
    """主函數"""
    logger.info("🚀 Starting improved quantum training and testing demo...")
    
    if not QUANTUM_AVAILABLE:
        logger.error("❌ Quantum computing libraries not available. Please install qiskit and pennylane.")
        return
    
    # 1. 創建樣本數據
    price_data = create_sample_data(days=500)
    
    # 2. 特徵工程
    feature_engineer = QuantumFeatureEngineer(n_features=8)
    X = feature_engineer.create_quantum_features(price_data)
    
    # 3. 創建目標變量
    returns = price_data['close'].pct_change()
    y = (abs(returns) > 0.02).astype(int)
    
    # 對齊數據
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y.iloc[:min_len].values
    
    logger.info(f"Data prepared: X={X.shape}, y={y.shape}")
    logger.info(f"Target distribution: {np.bincount(y)}")
    
    # 4. 時間序列分割
    test_size = int(len(X) * 0.2)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 5. 模型比較
    comparator = QuantumModelComparator()
    results = comparator.compare_models(X_train, y_train, X_test, y_test)
    
    # 6. 創建比較報告
    comparator.create_comparison_report()
    
    # 7. 打印結果摘要
    logger.info("\n" + "="*60)
    logger.info("QUANTUM MODEL COMPARISON RESULTS")
    logger.info("="*60)
    
    for model_name, result in results.items():
        if 'error' in result:
            logger.info(f"{model_name}: ❌ {result['error']}")
        else:
            logger.info(f"{model_name}:")
            logger.info(f"  Train Accuracy: {result['train_accuracy']:.4f}")
            logger.info(f"  Test Accuracy: {result['test_accuracy']:.4f}")
            logger.info(f"  Training Time: {result['training_time']:.2f}s")
            logger.info(f"  Overfitting: {result['train_accuracy'] - result['test_accuracy']:.4f}")
    
    logger.info("\n🎉 Quantum training and testing demo completed!")
    logger.info("📁 Results saved to reports/quantum_training_evaluation/")

if __name__ == "__main__":
    main()
