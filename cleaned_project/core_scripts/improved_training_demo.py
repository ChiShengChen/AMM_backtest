"""
改進的訓練與測試演示腳本
展示最佳實踐的機器學習模型訓練和測試流程
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

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """改進的特徵工程器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """創建穩健的特徵"""
        logger.info("Creating advanced features...")
        
        features = pd.DataFrame(index=price_data.index)
        
        # 1. 價格特徵
        features['returns'] = price_data['close'].pct_change()
        features['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
        features['price_change'] = price_data['close'] - price_data['open']
        features['price_range'] = price_data['high'] - price_data['low']
        features['price_position'] = (price_data['close'] - price_data['low']) / (price_data['high'] - price_data['low'])
        
        # 2. 移動平均線
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = price_data['close'].rolling(window).mean()
            features[f'price_vs_sma_{window}'] = price_data['close'] / features[f'sma_{window}'] - 1
        
        # 3. 技術指標
        features['rsi'] = self._calculate_rsi(price_data['close'])
        features['macd'] = self._calculate_macd(price_data['close'])
        features['bollinger_upper'], features['bollinger_lower'] = self._calculate_bollinger_bands(price_data['close'])
        features['bollinger_position'] = (price_data['close'] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'])
        
        # 4. 波動率特徵
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features[f'volatility_{window}'].rolling(50).mean()
        
        # 5. 成交量特徵
        features['volume_ma'] = price_data['volume'].rolling(20).mean()
        features['volume_ratio'] = price_data['volume'] / features['volume_ma']
        features['volume_price_trend'] = features['volume_ratio'] * features['returns']
        
        # 6. 滯後特徵
        for lag in [1, 2, 3, 5, 10]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'volatility_lag_{lag}'] = features['volatility_20'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)
        
        # 7. 時間特徵
        features['hour'] = price_data.index.hour
        features['day_of_week'] = price_data.index.dayofweek
        features['month'] = price_data.index.month
        
        # 8. 特徵選擇（移除高度相關的特徵）
        features = self._remove_correlated_features(features)
        
        # 保存特徵名稱
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Created {len(features.columns)} features")
        return features.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """計算RSI指標"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """計算MACD指標"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """計算布林帶"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def _remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """移除高度相關的特徵"""
        corr_matrix = features.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
            features = features.drop(columns=to_drop)
        
        return features

class AdvancedMLTrainer:
    """改進的ML訓練器"""
    
    def __init__(self, 
                 feature_engineer: AdvancedFeatureEngineer,
                 models_dir: str = "models",
                 test_size: float = 0.2,
                 n_splits: int = 5):
        self.feature_engineer = feature_engineer
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.test_size = test_size
        self.n_splits = n_splits
        
    def prepare_training_data(self, 
                            price_data: pd.DataFrame,
                            rebalance_threshold: float = 0.02) -> tuple:
        """準備訓練數據"""
        logger.info("Preparing training data...")
        
        # 創建特徵
        X = self.feature_engineer.create_features(price_data)
        
        # 創建目標變量（再平衡決策）
        returns = price_data['close'].pct_change()
        y_rebalance = (abs(returns) > rebalance_threshold).astype(int)
        
        # 對齊數據
        common_index = X.index.intersection(y_rebalance.index)
        X = X.loc[common_index]
        y_rebalance = y_rebalance.loc[common_index]
        
        logger.info(f"Training data shape: X={X.shape}, y={y_rebalance.shape}")
        logger.info(f"Rebalance rate: {y_rebalance.mean():.4f}")
        
        return X, y_rebalance
    
    def train_with_time_series_cv(self, 
                                price_data: pd.DataFrame,
                                model_type: str = 'random_forest') -> dict:
        """使用時間序列交叉驗證訓練模型"""
        logger.info(f"Training {model_type} model with time series CV...")
        
        # 準備數據
        X, y = self.prepare_training_data(price_data)
        
        # 時間序列交叉驗證
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 訓練模型
            model = self._create_model(model_type)
            model.fit(X_train, y_train)
            
            # 驗證
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # 計算多個指標
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'train_size': len(X_train),
                'val_size': len(X_val)
            }
            
            fold_results.append(fold_result)
            cv_scores.append(accuracy)
            
            logger.info(f"Fold {fold+1}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        
        # 最終模型訓練
        final_model = self._create_model(model_type)
        final_model.fit(X, y)
        
        # 計算交叉驗證統計
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"CV Results: Mean={cv_mean:.4f} ± {cv_std:.4f}")
        
        return {
            'model': final_model,
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'fold_results': fold_results,
            'feature_names': self.feature_engineer.feature_names
        }
    
    def _create_model(self, model_type: str):
        """創建模型"""
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        elif model_type == 'svm':
            return SVC(
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class ComprehensiveEvaluator:
    """綜合評估器"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, 
                      model, 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series,
                      price_data: pd.DataFrame) -> dict:
        """綜合模型評估"""
        logger.info("Evaluating model comprehensively...")
        
        # 1. 預測
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 2. 分類指標
        classification_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # 3. 特徵重要性
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        # 4. 交易性能模擬
        trading_metrics = self._simulate_trading_performance(y_pred, y_test, price_data)
        
        # 5. 穩定性測試
        stability_metrics = self._test_model_stability(model, X_test, y_test)
        
        evaluation_result = {
            'classification': classification_metrics,
            'feature_importance': feature_importance,
            'trading': trading_metrics,
            'stability': stability_metrics,
            'predictions': {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_test': y_test
            }
        }
        
        self.evaluation_results = evaluation_result
        return evaluation_result
    
    def _simulate_trading_performance(self, y_pred: np.ndarray, y_test: pd.Series, price_data: pd.DataFrame) -> dict:
        """模擬交易性能"""
        # 簡化的交易模擬
        portfolio_value = 10000
        positions = []
        
        for i, (pred, actual) in enumerate(zip(y_pred, y_test)):
            if pred == 1:  # 預測需要再平衡
                # 簡化的收益計算
                if i > 0:
                    price_change = price_data['close'].iloc[i] / price_data['close'].iloc[i-1] - 1
                    portfolio_value *= (1 + price_change * 0.1)  # 假設10%的價格變動轉化為收益
                    positions.append(portfolio_value)
        
        if positions:
            total_return = (positions[-1] - 10000) / 10000 * 100
            max_value = max(positions)
            min_value = min(positions)
            max_drawdown = (max_value - min_value) / max_value * 100
        else:
            total_return = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': portfolio_value,
            'num_trades': sum(y_pred)
        }
    
    def _test_model_stability(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """測試模型穩定性"""
        # 使用交叉驗證測試穩定性
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def create_evaluation_report(self, save_path: str = "reports/training_evaluation"):
        """創建評估報告"""
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return
        
        # 創建目錄
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 1. 分類性能圖表
        self._plot_classification_performance(save_path)
        
        # 2. 特徵重要性圖表
        if self.evaluation_results['feature_importance']:
            self._plot_feature_importance(save_path)
        
        # 3. 預測分布圖表
        self._plot_prediction_distribution(save_path)
        
        logger.info(f"Evaluation report saved to {save_path}")
    
    def _plot_classification_performance(self, save_path: str):
        """繪製分類性能圖表"""
        metrics = self.evaluation_results['classification']
        
        plt.figure(figsize=(12, 8))
        
        # 1. 指標對比
        plt.subplot(2, 2, 1)
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        plt.bar(metric_names, metric_values)
        plt.title('Classification Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        
        # 2. ROC曲線
        plt.subplot(2, 2, 2)
        from sklearn.metrics import roc_curve
        y_test = self.evaluation_results['predictions']['y_test']
        y_pred_proba = self.evaluation_results['predictions']['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc_roc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # 3. 預測概率分布
        plt.subplot(2, 2, 3)
        plt.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='No Rebalance', density=True)
        plt.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Rebalance', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        
        # 4. 混淆矩陣
        plt.subplot(2, 2, 4)
        from sklearn.metrics import confusion_matrix
        y_pred = self.evaluation_results['predictions']['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/classification_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, save_path: str):
        """繪製特徵重要性圖表"""
        feature_importance = self.evaluation_results['feature_importance']
        
        # 取前20個最重要的特徵
        top_features = dict(list(feature_importance.items())[:20])
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), list(top_features.values()))
        plt.yticks(range(len(top_features)), list(top_features.keys()))
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_distribution(self, save_path: str):
        """繪製預測分布圖表"""
        y_pred = self.evaluation_results['predictions']['y_pred']
        y_test = self.evaluation_results['predictions']['y_test']
        
        plt.figure(figsize=(10, 6))
        
        # 預測準確性隨時間變化
        accuracy_window = 50
        rolling_accuracy = []
        
        for i in range(accuracy_window, len(y_pred)):
            window_pred = y_pred[i-accuracy_window:i]
            window_test = y_test.iloc[i-accuracy_window:i]
            accuracy = accuracy_score(window_test, window_pred)
            rolling_accuracy.append(accuracy)
        
        plt.plot(rolling_accuracy)
        plt.title('Rolling Accuracy (50-sample window)')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/prediction_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_sample_data(days: int = 1000) -> pd.DataFrame:
    """創建樣本數據"""
    logger.info(f"Creating {days} days of sample data...")
    
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    # 模擬價格數據
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # 日收益率
    prices = 100 * np.exp(np.cumsum(returns))  # 價格序列
    
    # 創建OHLCV數據
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # 模擬日內波動
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
    logger.info("🚀 Starting improved training and testing demo...")
    
    # 1. 創建樣本數據
    price_data = create_sample_data(days=1000)
    
    # 2. 初始化組件
    feature_engineer = AdvancedFeatureEngineer()
    trainer = AdvancedMLTrainer(feature_engineer)
    evaluator = ComprehensiveEvaluator()
    
    # 3. 訓練多個模型
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    results = {}
    
    for model_type in model_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type} model...")
        logger.info(f"{'='*50}")
        
        # 訓練模型
        training_result = trainer.train_with_time_series_cv(price_data, model_type)
        
        # 準備測試數據
        X, y = trainer.prepare_training_data(price_data)
        test_size = int(len(X) * 0.2)
        X_test, y_test = X.iloc[-test_size:], y.iloc[-test_size:]
        
        # 評估模型
        evaluation_result = evaluator.evaluate_model(
            training_result['model'], X_test, y_test, price_data.iloc[-test_size:]
        )
        
        results[model_type] = {
            'training': training_result,
            'evaluation': evaluation_result
        }
        
        # 創建評估報告
        evaluator.create_evaluation_report(f"reports/training_evaluation/{model_type}")
    
    # 4. 模型比較
    logger.info(f"\n{'='*50}")
    logger.info("Model Comparison Results")
    logger.info(f"{'='*50}")
    
    comparison_data = []
    for model_type, result in results.items():
        eval_metrics = result['evaluation']['classification']
        trading_metrics = result['evaluation']['trading']
        
        comparison_data.append({
            'Model': model_type,
            'Accuracy': eval_metrics['accuracy'],
            'F1_Score': eval_metrics['f1_score'],
            'AUC': eval_metrics['auc_roc'],
            'Total_Return': trading_metrics['total_return'],
            'Max_Drawdown': trading_metrics['max_drawdown'],
            'CV_Mean': result['training']['cv_mean'],
            'CV_Std': result['training']['cv_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4))
    
    # 保存比較結果
    comparison_df.to_csv('reports/training_evaluation/model_comparison.csv', index=False)
    
    # 5. 創建比較圖表
    plt.figure(figsize=(15, 10))
    
    # 性能指標比較
    plt.subplot(2, 3, 1)
    comparison_df.set_index('Model')[['Accuracy', 'F1_Score', 'AUC']].plot(kind='bar', ax=plt.gca())
    plt.title('Classification Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    
    # 交易性能比較
    plt.subplot(2, 3, 2)
    comparison_df.set_index('Model')[['Total_Return', 'Max_Drawdown']].plot(kind='bar', ax=plt.gca())
    plt.title('Trading Performance Comparison')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.legend()
    
    # 交叉驗證結果
    plt.subplot(2, 3, 3)
    plt.bar(comparison_df['Model'], comparison_df['CV_Mean'], yerr=comparison_df['CV_Std'], capsize=5)
    plt.title('Cross-Validation Results')
    plt.ylabel('CV Accuracy')
    plt.xticks(rotation=45)
    
    # 綜合評分
    plt.subplot(2, 3, 4)
    comparison_df['Composite_Score'] = (
        comparison_df['Accuracy'] * 0.3 +
        comparison_df['F1_Score'] * 0.3 +
        comparison_df['AUC'] * 0.2 +
        comparison_df['Total_Return'] / 100 * 0.2
    )
    plt.bar(comparison_df['Model'], comparison_df['Composite_Score'])
    plt.title('Composite Performance Score')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    
    # 特徵重要性比較（使用最佳模型）
    best_model_type = comparison_df.loc[comparison_df['Composite_Score'].idxmax(), 'Model']
    best_result = results[best_model_type]
    
    if best_result['evaluation']['feature_importance']:
        plt.subplot(2, 3, 5)
        feature_importance = best_result['evaluation']['feature_importance']
        top_features = dict(list(feature_importance.items())[:10])
        plt.barh(range(len(top_features)), list(top_features.values()))
        plt.yticks(range(len(top_features)), list(top_features.keys()))
        plt.title(f'Top 10 Features - {best_model_type.title()}')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
    
    # 模型穩定性
    plt.subplot(2, 3, 6)
    stability_data = []
    for model_type, result in results.items():
        stability = result['evaluation']['stability']
        stability_data.append({
            'Model': model_type,
            'CV_Mean': stability['cv_mean'],
            'CV_Std': stability['cv_std']
        })
    
    stability_df = pd.DataFrame(stability_data)
    plt.bar(stability_df['Model'], stability_df['CV_Mean'], yerr=stability_df['CV_Std'], capsize=5)
    plt.title('Model Stability (CV Results)')
    plt.ylabel('CV Accuracy')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('reports/training_evaluation/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("\n🎉 Improved training and testing demo completed!")
    logger.info(f"📁 Results saved to reports/training_evaluation/")
    logger.info(f"🏆 Best model: {best_model_type}")

if __name__ == "__main__":
    main()
