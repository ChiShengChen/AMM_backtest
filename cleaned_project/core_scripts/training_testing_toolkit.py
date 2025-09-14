"""
訓練與測試工具包
提供完整的機器學習模型訓練、測試和評估功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
import joblib
from typing import Dict, List, Tuple, Any, Optional
warnings.filterwarnings('ignore')

# 機器學習庫
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report,
                           mean_squared_error, mean_absolute_error)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 量子計算庫（可選）
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import ZZFeatureMap, TwoLocal
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.primitives import Sampler
    import pennylane as qml
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingTestingToolkit:
    """訓練與測試工具包主類"""
    
    def __init__(self, 
                 models_dir: str = "models",
                 reports_dir: str = "reports",
                 enable_quantum: bool = True):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.enable_quantum = enable_quantum and QUANTUM_AVAILABLE
        
        # 創建目錄
        self.models_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # 初始化組件
        self.feature_engineer = FeatureEngineer()
        self.trainer = MLTrainer()
        self.evaluator = ModelEvaluator()
        
        if self.enable_quantum:
            self.quantum_trainer = QuantumTrainer()
            logger.info("✅ Quantum computing enabled")
        else:
            logger.info("⚠️ Quantum computing disabled")
        
        self.results = {}
    
    def run_complete_pipeline(self, 
                            price_data: pd.DataFrame,
                            target_column: str = 'rebalance_signal',
                            model_types: List[str] = None,
                            test_size: float = 0.2,
                            cv_folds: int = 5) -> Dict[str, Any]:
        """運行完整的訓練測試管道"""
        logger.info("🚀 Starting complete training and testing pipeline...")
        
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
            if self.enable_quantum:
                model_types.extend(['qiskit_vqc', 'pennylane_qnn'])
        
        # 1. 數據準備
        logger.info("📊 Preparing data...")
        X, y = self._prepare_data(price_data, target_column)
        
        # 2. 數據分割
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        logger.info(f"Data split: Train={X_train.shape}, Test={X_test.shape}")
        
        # 3. 模型訓練和評估
        results = {}
        for model_type in model_types:
            logger.info(f"🔄 Training {model_type}...")
            
            try:
                if model_type in ['qiskit_vqc', 'pennylane_qnn'] and self.enable_quantum:
                    result = self._train_quantum_model(model_type, X_train, y_train, X_test, y_test)
                else:
                    result = self._train_classical_model(model_type, X_train, y_train, X_test, y_test, cv_folds)
                
                results[model_type] = result
                logger.info(f"✅ {model_type} completed: Test Accuracy = {result['test_accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {model_type} failed: {e}")
                results[model_type] = {'error': str(e)}
        
        # 4. 結果比較
        comparison = self._compare_models(results)
        
        # 5. 生成報告
        self._generate_reports(results, comparison)
        
        self.results = results
        return results
    
    def _prepare_data(self, price_data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """準備訓練數據"""
        # 創建特徵
        X = self.feature_engineer.create_features(price_data)
        
        # 創建目標變量
        if target_column not in price_data.columns:
            # 自動創建再平衡信號
            returns = price_data['close'].pct_change()
            y = (abs(returns) > 0.02).astype(int)
        else:
            y = price_data[target_column]
        
        # 對齊數據
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y.iloc[:min_len].values if hasattr(y, 'iloc') else y[:min_len]
        
        return X, y
    
    def _train_classical_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray, cv_folds: int) -> Dict[str, Any]:
        """訓練經典機器學習模型"""
        # 訓練模型
        model, training_result = self.trainer.train_model(
            model_type, X_train, y_train, cv_folds=cv_folds
        )
        
        # 測試模型
        test_result = self.evaluator.evaluate_model(model, X_test, y_test)
        
        # 合併結果
        result = {
            'model': model,
            'model_type': model_type,
            'training': training_result,
            'test_accuracy': test_result['accuracy'],
            'test_precision': test_result['precision'],
            'test_recall': test_result['recall'],
            'test_f1': test_result['f1_score'],
            'test_auc': test_result['auc'],
            'training_time': training_result['training_time'],
            'cv_mean': training_result['cv_mean'],
            'cv_std': training_result['cv_std']
        }
        
        return result
    
    def _train_quantum_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """訓練量子機器學習模型"""
        # 準備量子特徵
        X_train_q = self.feature_engineer.create_quantum_features(X_train)
        X_test_q = self.feature_engineer.create_quantum_features(X_test)
        
        # 訓練量子模型
        model, training_result = self.quantum_trainer.train_model(
            model_type, X_train_q, y_train
        )
        
        # 測試模型
        y_pred = model.predict(X_test_q)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        result = {
            'model': model,
            'model_type': model_type,
            'training': training_result,
            'test_accuracy': test_accuracy,
            'training_time': training_result['training_time']
        }
        
        return result
    
    def _compare_models(self, results: Dict[str, Any]) -> pd.DataFrame:
        """比較模型性能"""
        comparison_data = []
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
            
            # 計算綜合評分
            composite_score = (
                result['test_accuracy'] * 0.4 +
                result.get('test_f1', 0) * 0.3 +
                result.get('test_auc', 0) * 0.2 +
                (1 - result['training_time'] / 100) * 0.1  # 時間懲罰
            )
            
            comparison_data.append({
                'Model': model_name,
                'Test_Accuracy': result['test_accuracy'],
                'Test_F1': result.get('test_f1', 0),
                'Test_AUC': result.get('test_auc', 0),
                'Training_Time': result['training_time'],
                'CV_Mean': result.get('cv_mean', 0),
                'CV_Std': result.get('cv_std', 0),
                'Composite_Score': composite_score
            })
        
        return pd.DataFrame(comparison_data)
    
    def _generate_reports(self, results: Dict[str, Any], comparison: pd.DataFrame):
        """生成報告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = self.reports_dir / f"training_report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # 1. 保存比較結果
        comparison.to_csv(report_dir / "model_comparison.csv", index=False)
        
        # 2. 生成比較圖表
        self._plot_model_comparison(comparison, report_dir)
        
        # 3. 生成詳細報告
        self._generate_detailed_report(results, comparison, report_dir)
        
        logger.info(f"📁 Reports saved to {report_dir}")
    
    def _plot_model_comparison(self, comparison: pd.DataFrame, report_dir: Path):
        """繪製模型比較圖表"""
        plt.figure(figsize=(15, 10))
        
        # 1. 準確率比較
        plt.subplot(2, 3, 1)
        plt.bar(comparison['Model'], comparison['Test_Accuracy'], alpha=0.8)
        plt.title('Test Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. 訓練時間比較
        plt.subplot(2, 3, 2)
        plt.bar(comparison['Model'], comparison['Training_Time'], alpha=0.8, color='orange')
        plt.title('Training Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. 綜合評分
        plt.subplot(2, 3, 3)
        plt.bar(comparison['Model'], comparison['Composite_Score'], alpha=0.8, color='green')
        plt.title('Composite Score Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. 性能雷達圖
        plt.subplot(2, 3, 4)
        metrics = ['Test_Accuracy', 'Test_F1', 'Test_AUC']
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 閉合圓圈
        
        for _, row in comparison.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # 閉合圓圈
            
            plt.polar(angles, values, 'o-', linewidth=2, label=row['Model'])
            plt.fill(angles, values, alpha=0.25)
        
        plt.xticks(angles[:-1], metrics)
        plt.title('Performance Radar Chart')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 5. 準確率 vs 訓練時間散點圖
        plt.subplot(2, 3, 5)
        plt.scatter(comparison['Training_Time'], comparison['Test_Accuracy'], 
                   s=100, alpha=0.7)
        
        for i, model in enumerate(comparison['Model']):
            plt.annotate(model, 
                        (comparison['Training_Time'].iloc[i], comparison['Test_Accuracy'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy vs Training Time')
        plt.grid(True, alpha=0.3)
        
        # 6. 交叉驗證結果
        plt.subplot(2, 3, 6)
        plt.bar(comparison['Model'], comparison['CV_Mean'], 
               yerr=comparison['CV_Std'], capsize=5, alpha=0.8, color='purple')
        plt.title('Cross-Validation Results')
        plt.ylabel('CV Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(report_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_detailed_report(self, results: Dict[str, Any], comparison: pd.DataFrame, report_dir: Path):
        """生成詳細報告"""
        report_content = f"""
# 訓練與測試詳細報告

## 執行時間
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 模型比較結果

{comparison.to_string(index=False)}

## 最佳模型
{comparison.loc[comparison['Composite_Score'].idxmax(), 'Model']}

## 詳細結果

"""
        
        for model_name, result in results.items():
            if 'error' in result:
                report_content += f"""
### {model_name}
❌ 錯誤: {result['error']}
"""
            else:
                report_content += f"""
### {model_name}
- 測試準確率: {result['test_accuracy']:.4f}
- 測試F1分數: {result.get('test_f1', 'N/A')}
- 測試AUC: {result.get('test_auc', 'N/A')}
- 訓練時間: {result['training_time']:.2f}秒
- 交叉驗證均值: {result.get('cv_mean', 'N/A')}
- 交叉驗證標準差: {result.get('cv_std', 'N/A')}
"""
        
        with open(report_dir / "detailed_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)

class FeatureEngineer:
    """特徵工程器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.quantum_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
        self.feature_names = []
    
    def create_features(self, price_data: pd.DataFrame) -> np.ndarray:
        """創建經典ML特徵"""
        features = pd.DataFrame(index=price_data.index)
        
        # 基本特徵
        features['returns'] = price_data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['volume_ratio'] = price_data['volume'] / price_data['volume'].rolling(20).mean()
        
        # 技術指標
        features['rsi'] = self._calculate_rsi(price_data['close'])
        features['macd'] = self._calculate_macd(price_data['close'])
        
        # 滯後特徵
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        # 處理缺失值
        features = features.fillna(features.mean())
        
        # 標準化
        features_scaled = self.scaler.fit_transform(features)
        
        self.feature_names = features.columns.tolist()
        return features_scaled
    
    def create_quantum_features(self, features: np.ndarray) -> np.ndarray:
        """創建量子ML特徵"""
        # 限制特徵數量
        if features.shape[1] > 8:
            features = features[:, :8]
        
        # 標準化到量子範圍
        features_scaled = self.quantum_scaler.fit_transform(features)
        return features_scaled
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """計算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """計算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd / prices

class MLTrainer:
    """經典ML訓練器"""
    
    def __init__(self):
        self.models = {}
    
    def train_model(self, model_type: str, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Tuple[Any, Dict]:
        """訓練經典ML模型"""
        start_time = datetime.now()
        
        # 創建模型
        model = self._create_model(model_type)
        
        # 交叉驗證
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        
        # 訓練最終模型
        model.fit(X, y)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return model, {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time
        }
    
    def _create_model(self, model_type: str):
        """創建模型"""
        if model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'svm':
            return SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class QuantumTrainer:
    """量子ML訓練器"""
    
    def __init__(self):
        if not QUANTUM_AVAILABLE:
            raise ImportError("Quantum libraries not available")
    
    def train_model(self, model_type: str, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict]:
        """訓練量子模型"""
        start_time = datetime.now()
        
        if model_type == 'qiskit_vqc':
            model = self._train_qiskit_vqc(X, y)
        elif model_type == 'pennylane_qnn':
            model = self._train_pennylane_qnn(X, y)
        else:
            raise ValueError(f"Unknown quantum model type: {model_type}")
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return model, {
            'training_time': training_time
        }
    
    def _train_qiskit_vqc(self, X: np.ndarray, y: np.ndarray):
        """訓練VQE Classifier"""
        feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=1)
        ansatz = TwoLocal(X.shape[1], ['ry', 'rz'], 'cz', reps=2)
        
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=50),
            sampler=Sampler()
        )
        
        vqc.fit(X, y)
        return vqc
    
    def _train_pennylane_qnn(self, X: np.ndarray, y: np.ndarray):
        """訓練QNN"""
        n_qubits = min(4, X.shape[1])
        device = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(device)
        def circuit(features, weights):
            for i, feature in enumerate(features[:n_qubits]):
                qml.RY(feature, wires=i)
            
            for layer in range(2):
                for i in range(n_qubits):
                    qml.RY(weights[layer][i], wires=i)
                    qml.RZ(weights[layer][i + n_qubits], wires=i)
                
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return qml.expval(qml.PauliZ(0))
        
        # 簡化訓練
        weights = np.random.uniform(0, 2*np.pi, (2, 2*n_qubits))
        
        class PennyLaneModel:
            def __init__(self, circuit, weights):
                self.circuit = circuit
                self.weights = weights
            
            def predict(self, X):
                predictions = []
                for i in range(len(X)):
                    pred = self.circuit(X[i], self.weights)
                    predictions.append(1 if pred > 0 else 0)
                return np.array(predictions)
        
        return PennyLaneModel(circuit, weights)

class ModelEvaluator:
    """模型評估器"""
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """評估模型"""
        y_pred = model.predict(X_test)
        
        # 計算指標
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # 計算AUC（如果模型支持概率預測）
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

def create_sample_data(days: int = 1000) -> pd.DataFrame:
    """創建樣本數據"""
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)
    prices = 100 * np.exp(np.cumsum(returns))
    
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
    return df

def main():
    """主函數 - 演示工具包使用"""
    logger.info("🚀 Starting Training and Testing Toolkit Demo...")
    
    # 1. 創建樣本數據
    price_data = create_sample_data(days=1000)
    
    # 2. 初始化工具包
    toolkit = TrainingTestingToolkit(enable_quantum=QUANTUM_AVAILABLE)
    
    # 3. 運行完整管道
    results = toolkit.run_complete_pipeline(
        price_data=price_data,
        model_types=['random_forest', 'gradient_boosting', 'logistic_regression'] + 
                   (['qiskit_vqc', 'pennylane_qnn'] if QUANTUM_AVAILABLE else [])
    )
    
    # 4. 打印結果摘要
    logger.info("\n" + "="*60)
    logger.info("TRAINING AND TESTING TOOLKIT RESULTS")
    logger.info("="*60)
    
    for model_name, result in results.items():
        if 'error' in result:
            logger.info(f"{model_name}: ❌ {result['error']}")
        else:
            logger.info(f"{model_name}: ✅ Test Accuracy = {result['test_accuracy']:.4f}")
    
    logger.info("\n🎉 Training and Testing Toolkit Demo completed!")
    logger.info("📁 Check the reports directory for detailed results")

if __name__ == "__main__":
    main()
