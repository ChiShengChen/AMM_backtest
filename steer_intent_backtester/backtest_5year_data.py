"""
5年日線數據回測腳本 - Steer Intent Backtester
測試量子/經典ML策略在真實數據上的表現
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_dir: str = "../amm-rebalance-backtester/data/5year_daily") -> dict:
    """加載5年日線數據"""
    data_files = {
        'BTCUSDC': 'BTCUSDC_1d_20200905_20250903.csv',
        'ETHUSDC': 'ETHUSDC_1d_20200905_20250903.csv', 
        'USDCUSDT': 'USDCUSDT_1d_20200905_20250903.csv'
    }
    
    data = {}
    for symbol, filename in data_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} records from {df.index[0]} to {df.index[-1]}")
        else:
            logger.warning(f"File not found: {filepath}")
    
    return data

def prepare_data_for_backtest(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """為回測準備數據"""
    # 確保數據按時間排序
    df = df.sort_index()
    
    # 添加基本特徵
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # 移除NaN值
    df = df.dropna()
    
    logger.info(f"{symbol} prepared: {len(df)} records, price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    return df

def run_quantum_strategies(data: dict):
    """運行量子策略回測"""
    logger.info("=" * 60)
    logger.info("運行量子策略回測")
    logger.info("=" * 60)
    
    try:
        from steerbt.strategies import (
            QuantumBollingerStrategy,
            QuantumKeltnerStrategy, 
            QuantumHybridStrategy
        )
        
        results = {}
        
        for symbol, df in data.items():
            logger.info(f"\n測試 {symbol} 量子策略...")
            
            # 準備數據
            df_prepared = prepare_data_for_backtest(df.copy(), symbol)
            
            # 測試不同的量子策略
            strategies = {
                'QuantumBollinger': QuantumBollingerStrategy(
                    quantum_model_type="qnn_intent",
                    n_qubits=2,
                    n_layers=1,
                    feature_dim=4,
                    n=20,
                    k=2.0,
                    rebalance_cooldown_hours=24
                ),
                'QuantumKeltner': QuantumKeltnerStrategy(
                    quantum_model_type="qsvm_price",
                    n_qubits=2,
                    feature_dim=4,
                    n=20,
                    m=2.0,
                    rebalance_cooldown_hours=48
                ),
                'QuantumHybrid': QuantumHybridStrategy(
                    intent_model_type="qnn_intent",
                    price_model_type="qsvm_price",
                    n_qubits=2,
                    n_layers=1,
                    feature_dim=4,
                    intent_weight=0.6,
                    price_weight=0.4,
                    rebalance_cooldown_hours=24
                )
            }
            
            symbol_results = {}
            
            for strategy_name, strategy in strategies.items():
                logger.info(f"  測試 {strategy_name}...")
                
                # 模擬回測
                rebalance_decisions = []
                position_widths = []
                portfolio_values = []
                current_portfolio = 10000  # 初始資金
                
                for i in range(50, len(df_prepared)):
                    current_data = df_prepared.iloc[:i+1]
                    current_price = df_prepared['close'].iloc[i]
                    current_time = df_prepared.index[i]
                    
                    # 檢查是否需要再平衡
                    should_rebalance = strategy.should_rebalance(
                        current_price=current_price,
                        current_time=current_time,
                        price_data=current_data
                    )
                    
                    # 計算位置範圍
                    ranges, liquidities = strategy.calculate_range(
                        price_data=current_data,
                        current_price=current_price,
                        portfolio_value=current_portfolio
                    )
                    
                    rebalance_decisions.append(should_rebalance)
                    
                    if ranges:
                        width_pct = (ranges[0][1] - ranges[0][0]) / current_price
                        position_widths.append(width_pct)
                        
                        # 簡化的投資組合價值計算
                        if should_rebalance:
                            # 假設再平衡時有0.1%的交易成本
                            current_portfolio *= 0.999
                    else:
                        position_widths.append(0)
                    
                    portfolio_values.append(current_portfolio)
                
                # 計算策略統計
                total_rebalances = sum(rebalance_decisions)
                avg_width = np.mean(position_widths) if position_widths else 0
                final_portfolio = portfolio_values[-1] if portfolio_values else current_portfolio
                total_return = (final_portfolio - 10000) / 10000 * 100
                
                symbol_results[strategy_name] = {
                    'total_rebalances': total_rebalances,
                    'avg_position_width': avg_width,
                    'final_portfolio': final_portfolio,
                    'total_return_pct': total_return,
                    'strategy_info': strategy.get_strategy_info()
                }
                
                logger.info(f"    {strategy_name}: 再平衡 {total_rebalances} 次, "
                           f"平均寬度 {avg_width:.3f}, 總收益 {total_return:.2f}%")
            
            results[symbol] = symbol_results
        
        return results
        
    except ImportError as e:
        logger.error(f"量子策略導入失敗: {e}")
        return {}

def run_classic_ml_strategies(data: dict):
    """運行經典ML策略回測"""
    logger.info("=" * 60)
    logger.info("運行經典ML策略回測")
    logger.info("=" * 60)
    
    try:
        from steerbt.strategies import (
            MLBollingerStrategy,
            MLKeltnerStrategy, 
            MLHybridStrategy
        )
        
        results = {}
        
        for symbol, df in data.items():
            logger.info(f"\n測試 {symbol} 經典ML策略...")
            
            # 準備數據
            df_prepared = prepare_data_for_backtest(df.copy(), symbol)
            
            # 測試不同的經典ML策略
            strategies = {
                'MLBollinger': MLBollingerStrategy(
                    n=20,
                    k=2.0,
                    rebalance_cooldown_hours=24
                ),
                'MLKeltner': MLKeltnerStrategy(
                    n=20,
                    m=2.0,
                    rebalance_cooldown_hours=48
                ),
                'MLHybrid': MLHybridStrategy(
                    intent_weight=0.6,
                    price_weight=0.4,
                    rebalance_cooldown_hours=24
                )
            }
            
            symbol_results = {}
            
            for strategy_name, strategy in strategies.items():
                logger.info(f"  測試 {strategy_name}...")
                
                # 模擬回測
                rebalance_decisions = []
                position_widths = []
                portfolio_values = []
                current_portfolio = 10000  # 初始資金
                
                for i in range(50, len(df_prepared)):
                    current_data = df_prepared.iloc[:i+1]
                    current_price = df_prepared['close'].iloc[i]
                    current_time = df_prepared.index[i]
                    
                    # 檢查是否需要再平衡
                    should_rebalance = strategy.should_rebalance(
                        current_price=current_price,
                        current_time=current_time,
                        price_data=current_data
                    )
                    
                    # 計算位置範圍
                    ranges, liquidities = strategy.calculate_range(
                        price_data=current_data,
                        current_price=current_price,
                        portfolio_value=current_portfolio
                    )
                    
                    rebalance_decisions.append(should_rebalance)
                    
                    if ranges:
                        width_pct = (ranges[0][1] - ranges[0][0]) / current_price
                        position_widths.append(width_pct)
                        
                        # 簡化的投資組合價值計算
                        if should_rebalance:
                            # 假設再平衡時有0.1%的交易成本
                            current_portfolio *= 0.999
                    else:
                        position_widths.append(0)
                    
                    portfolio_values.append(current_portfolio)
                
                # 計算策略統計
                total_rebalances = sum(rebalance_decisions)
                avg_width = np.mean(position_widths) if position_widths else 0
                final_portfolio = portfolio_values[-1] if portfolio_values else current_portfolio
                total_return = (final_portfolio - 10000) / 10000 * 100
                
                symbol_results[strategy_name] = {
                    'total_rebalances': total_rebalances,
                    'avg_position_width': avg_width,
                    'final_portfolio': final_portfolio,
                    'total_return_pct': total_return,
                    'strategy_info': strategy.get_strategy_info()
                }
                
                logger.info(f"    {strategy_name}: 再平衡 {total_rebalances} 次, "
                           f"平均寬度 {avg_width:.3f}, 總收益 {total_return:.2f}%")
            
            results[symbol] = symbol_results
        
        return results
        
    except ImportError as e:
        logger.error(f"經典ML策略導入失敗: {e}")
        return {}

def create_comparison_report(quantum_results: dict, classic_results: dict):
    """創建比較報告"""
    logger.info("=" * 60)
    logger.info("創建比較報告")
    logger.info("=" * 60)
    
    # 創建結果目錄
    results_dir = "reports/5year_backtest"
    os.makedirs(results_dir, exist_ok=True)
    
    # 收集所有結果
    all_results = []
    
    for symbol in ['BTCUSDC', 'ETHUSDC', 'USDCUSDT']:
        if symbol in quantum_results:
            for strategy_name, result in quantum_results[symbol].items():
                all_results.append({
                    'Symbol': symbol,
                    'Strategy': strategy_name,
                    'Type': 'Quantum',
                    'Rebalances': result['total_rebalances'],
                    'Avg_Width': result['avg_position_width'],
                    'Final_Portfolio': result['final_portfolio'],
                    'Return_Pct': result['total_return_pct']
                })
        
        if symbol in classic_results:
            for strategy_name, result in classic_results[symbol].items():
                all_results.append({
                    'Symbol': symbol,
                    'Strategy': strategy_name,
                    'Type': 'Classic ML',
                    'Rebalances': result['total_rebalances'],
                    'Avg_Width': result['avg_position_width'],
                    'Final_Portfolio': result['final_portfolio'],
                    'Return_Pct': result['total_return_pct']
                })
    
    # 創建結果DataFrame
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        # 保存結果到CSV
        results_df.to_csv(f"{results_dir}/backtest_results.csv", index=False)
        logger.info(f"結果已保存到 {results_dir}/backtest_results.csv")
        
        # 創建比較圖表
        plt.figure(figsize=(15, 10))
        
        # 1. 收益比較
        plt.subplot(2, 2, 1)
        pivot_returns = results_df.pivot_table(
            values='Return_Pct', 
            index=['Symbol', 'Strategy'], 
            columns='Type', 
            fill_value=0
        )
        pivot_returns.plot(kind='bar', ax=plt.gca())
        plt.title('策略收益比較 (%)')
        plt.ylabel('總收益 (%)')
        plt.xticks(rotation=45)
        plt.legend(title='策略類型')
        
        # 2. 再平衡次數比較
        plt.subplot(2, 2, 2)
        pivot_rebalances = results_df.pivot_table(
            values='Rebalances', 
            index=['Symbol', 'Strategy'], 
            columns='Type', 
            fill_value=0
        )
        pivot_rebalances.plot(kind='bar', ax=plt.gca())
        plt.title('再平衡次數比較')
        plt.ylabel('再平衡次數')
        plt.xticks(rotation=45)
        plt.legend(title='策略類型')
        
        # 3. 平均位置寬度比較
        plt.subplot(2, 2, 3)
        pivot_width = results_df.pivot_table(
            values='Avg_Width', 
            index=['Symbol', 'Strategy'], 
            columns='Type', 
            fill_value=0
        )
        pivot_width.plot(kind='bar', ax=plt.gca())
        plt.title('平均位置寬度比較')
        plt.ylabel('平均寬度')
        plt.xticks(rotation=45)
        plt.legend(title='策略類型')
        
        # 4. 策略類型平均表現
        plt.subplot(2, 2, 4)
        type_avg = results_df.groupby('Type')['Return_Pct'].mean()
        type_avg.plot(kind='bar', ax=plt.gca(), color=['blue', 'red'])
        plt.title('策略類型平均收益')
        plt.ylabel('平均收益 (%)')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"比較圖表已保存到 {results_dir}/strategy_comparison.png")
        
        # 打印總結
        print("\n" + "=" * 80)
        print("5年數據回測結果總結")
        print("=" * 80)
        
        print("\n📊 策略收益排名:")
        top_strategies = results_df.nlargest(10, 'Return_Pct')[['Symbol', 'Strategy', 'Type', 'Return_Pct']]
        for idx, row in top_strategies.iterrows():
            print(f"  {row['Symbol']} - {row['Strategy']} ({row['Type']}): {row['Return_Pct']:.2f}%")
        
        print("\n🔄 再平衡頻率排名:")
        rebalance_ranking = results_df.nlargest(10, 'Rebalances')[['Symbol', 'Strategy', 'Type', 'Rebalances']]
        for idx, row in rebalance_ranking.iterrows():
            print(f"  {row['Symbol']} - {row['Strategy']} ({row['Type']}): {row['Rebalances']} 次")
        
        print("\n📈 策略類型平均表現:")
        type_summary = results_df.groupby('Type').agg({
            'Return_Pct': ['mean', 'std', 'count'],
            'Rebalances': 'mean'
        }).round(2)
        print(type_summary)
        
        print(f"\n💾 詳細結果已保存到: {results_dir}/")
        print("   - backtest_results.csv: 詳細數據")
        print("   - strategy_comparison.png: 比較圖表")

def main():
    """主函數"""
    logger.info("開始5年日線數據回測...")
    
    # 加載數據
    data = load_data()
    if not data:
        logger.error("沒有找到數據文件")
        return
    
    # 運行量子策略回測
    quantum_results = run_quantum_strategies(data)
    
    # 運行經典ML策略回測
    classic_results = run_classic_ml_strategies(data)
    
    # 創建比較報告
    create_comparison_report(quantum_results, classic_results)
    
    logger.info("回測完成！")

if __name__ == "__main__":
    main()
