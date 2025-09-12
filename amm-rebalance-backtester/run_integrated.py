#!/usr/bin/env python3
"""
簡化的整合回測腳本
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# 設置路徑
sys.path.append('/Users/michael/Desktop/Omnis_bt/steer_intent_backtester')
sys.path.append('/Users/michael/Desktop/Omnis_bt/amm-rebalance-backtester')

# 導入AMM系統
from src.core.engine import BacktestEngine
from src.io.loader import DataLoader
from src.opt.search import OptunaOptimizer
from src.reporting.plots import PlotGenerator
from src.reporting.tables import TableGenerator

logger = logging.getLogger(__name__)

class SteerStrategyWrapper:
    """Steer策略包裝器"""
    
    def __init__(self, name: str, strategy_class, **params):
        self.name = name
        self.strategy_class = strategy_class
        self.params = params
        self.strategy = None
        
    def initialize(self, price_data: pd.DataFrame):
        """初始化策略"""
        try:
            self.strategy = self.strategy_class(**self.params)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    def calculate_ranges(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        **kwargs
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """計算範圍"""
        if self.strategy is None:
            # 返回默認範圍
            width = current_price * 0.1
            return [(current_price - width/2, current_price + width/2)], [portfolio_value]
        
        try:
            # 轉換數據格式
            steer_data = self._convert_data(price_data)
            ranges, liquidities = self.strategy.calculate_range(
                steer_data, current_price, portfolio_value
            )
            return ranges, liquidities
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            width = current_price * 0.1
            return [(current_price - width/2, current_price + width/2)], [portfolio_value]
    
    def should_rebalance(
        self,
        current_price: float,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """判斷是否需要再平衡"""
        # 簡化邏輯：每小時檢查一次
        return True
    
    def _convert_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """轉換數據格式"""
        converted = pd.DataFrame()
        converted['timestamp'] = price_data.index
        converted['open'] = price_data['open']
        converted['high'] = price_data['high']
        converted['low'] = price_data['low']
        converted['close'] = price_data['close']
        converted['volume'] = price_data['volume']
        return converted

def create_steer_strategies():
    """創建Steer策略"""
    strategies = {}
    
    try:
        # 導入Steer策略
        from steerbt.strategies import (
            ClassicStrategy,
            ChannelMultiplierStrategy,
            BollingerStrategy,
            KeltnerStrategy,
            DonchianStrategy,
            StableStrategy,
            FluidStrategy
        )
        
        # 經典策略
        strategies['steer_classic'] = SteerStrategyWrapper(
            'steer_classic',
            ClassicStrategy,
            width_mode='percentage',
            width_value=0.1,
            trigger_mode='center_deviation',
            trigger_value=0.05,
            curve_type='linear'
        )
        
        # 通道倍數策略
        strategies['steer_channel'] = SteerStrategyWrapper(
            'steer_channel',
            ChannelMultiplierStrategy,
            width_pct=0.15,
            trigger_pct=0.08
        )
        
        # 布林帶策略
        strategies['steer_bollinger'] = SteerStrategyWrapper(
            'steer_bollinger',
            BollingerStrategy,
            period=20,
            std_multiplier=2.0,
            trigger_pct=0.1
        )
        
        # 肯特納通道策略
        strategies['steer_keltner'] = SteerStrategyWrapper(
            'steer_keltner',
            KeltnerStrategy,
            ema_period=20,
            atr_multiplier=2.0,
            trigger_pct=0.1
        )
        
        # 唐奇安通道策略
        strategies['steer_donchian'] = SteerStrategyWrapper(
            'steer_donchian',
            DonchianStrategy,
            period=20,
            width_multiplier=1.0,
            trigger_pct=0.1
        )
        
        # 穩定策略
        strategies['steer_stable'] = SteerStrategyWrapper(
            'steer_stable',
            StableStrategy,
            anchor_period=50,
            width_pct=0.2,
            num_bins=5,
            curve_type='gaussian'
        )
        
        # 流體策略
        strategies['steer_fluid'] = SteerStrategyWrapper(
            'steer_fluid',
            FluidStrategy,
            ideal_ratio=0.5,
            imbalance_threshold=0.1,
            rebalance_threshold=0.05
        )
        
    except ImportError as e:
        logger.error(f"Failed to import Steer strategies: {e}")
        logger.info("Continuing with AMM strategies only...")
    
    return strategies

def run_integrated_backtest(pool: str, frequency: str, study_name: str, n_trials: int = 30):
    """運行整合回測"""
    
    print(f"🚀 Starting integrated backtest for {pool}")
    print(f"📊 Study: {study_name}")
    print(f"🔬 Trials: {n_trials}")
    
    # 加載配置
    config_path = "configs/integrated_experiment.yaml"
    
    # 加載數據
    print("📈 Loading price data...")
    from src.io.schema import ValidationConfig
    config = ValidationConfig()
    loader = DataLoader("data", config)
    price_data, _ = loader.load_pool_data(pool, frequency)
    print(f"✅ Loaded {len(price_data)} data points")
    
    # 創建Steer策略
    print("🔧 Creating Steer strategies...")
    steer_strategies = create_steer_strategies()
    print(f"✅ Created {len(steer_strategies)} Steer strategies")
    
    # 初始化引擎
    print("⚙️ Initializing backtest engine...")
    engine = BacktestEngine(config_path)
    
    # 添加Steer策略到引擎
    for name, strategy in steer_strategies.items():
        try:
            if strategy.initialize(price_data):
                engine.add_strategy(name, strategy)
                print(f"✅ Added {name}")
            else:
                print(f"❌ Failed to add {name}")
        except Exception as e:
            print(f"❌ Error adding {name}: {e}")
    
    # 運行優化
    print("🎯 Running optimization...")
    optimizer = OptunaOptimizer(engine)
    best_params = optimizer.optimize(
        study_name=study_name,
        n_trials=n_trials
    )
    
    print(f"🏆 Best parameters: {best_params}")
    
    # 運行最終評估
    print("📊 Running final evaluation...")
    results = engine.run_full_evaluation(best_params)
    
    # 生成報告
    print("📋 Generating reports...")
    # 生成圖表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pool_dir = f"reports/figs/{pool.lower()}"
    os.makedirs(pool_dir, exist_ok=True)
    
    plot_generator = PlotGenerator()
    plot_generator.plot_equity_curves(results, f"{pool_dir}/{pool}_equity_curves_{timestamp}.png")
    plot_generator.plot_apr_mdd_scatter(results, f"{pool_dir}/{pool}_apr_mdd_scatter_{timestamp}.png")
    plot_generator.plot_fee_vs_price_pnl(results, f"{pool_dir}/{pool}_fee_vs_price_pnl_{timestamp}.png")
    plot_generator.plot_sensitivity_heatmap(results, f"{pool_dir}/{pool}_sensitivity_heatmap_{timestamp}.png")
    plot_generator.plot_gas_frequency_contour(results, f"{pool_dir}/{pool}_gas_frequency_contour_{timestamp}.png")
    plot_generator.plot_il_curve(results, f"{pool_dir}/{pool}_il_curve_{timestamp}.png")
    plot_generator.plot_lvr_estimates(results, f"{pool_dir}/{pool}_lvr_estimates_{timestamp}.png")
    
    table_generator = TableGenerator()
    table_generator.generate_summary_tables(results)
    
    # 生成整合報告
    generate_integrated_report(results, pool, study_name, steer_strategies)
    
    print("✅ Integrated backtest completed!")
    print(f"📁 Results saved to results/ and reports/figs/{pool}/")
    
    return results

def generate_integrated_report(results: Dict, pool: str, study_name: str, steer_strategies: Dict):
    """生成整合報告"""
    
    report_path = f"results/integrated_report_{study_name}.txt"
    
    with open(report_path, 'w') as f:
        f.write("Integrated AMM + Steer Strategies Backtest Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Pool: {pool}\n")
        f.write(f"Study: {study_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("STRATEGY PERFORMANCE COMPARISON\n")
        f.write("-" * 40 + "\n\n")
        
        # 獲取策略結果
        strategy_results = results.get('strategy_results', {})
        
        if strategy_results:
            # 按APR排序
            sorted_strategies = sorted(
                strategy_results.items(),
                key=lambda x: x[1].get('apr', 0),
                reverse=True
            )
            
            f.write("Ranking by APR:\n")
            f.write("-" * 20 + "\n")
            for i, (name, metrics) in enumerate(sorted_strategies, 1):
                f.write(f"{i:2d}. {name:<25} APR: {metrics.get('apr', 0):6.2f}% "
                       f"MDD: {metrics.get('mdd', 0):6.2f}% "
                       f"Sharpe: {metrics.get('sharpe', 0):5.2f}\n")
            
            f.write("\n")
            
            # 按MDD排序
            sorted_by_mdd = sorted(
                strategy_results.items(),
                key=lambda x: x[1].get('mdd', 100)
            )
            
            f.write("Ranking by MDD (Lowest Risk):\n")
            f.write("-" * 30 + "\n")
            for i, (name, metrics) in enumerate(sorted_by_mdd, 1):
                f.write(f"{i:2d}. {name:<25} MDD: {metrics.get('mdd', 0):6.2f}% "
                       f"APR: {metrics.get('apr', 0):6.2f}% "
                       f"Sharpe: {metrics.get('sharpe', 0):5.2f}\n")
            
            f.write("\n")
            
            # 策略類型分析
            f.write("STRATEGY TYPE ANALYSIS\n")
            f.write("-" * 25 + "\n")
            
            amm_strategies = {k: v for k, v in strategy_results.items() if not k.startswith('steer_')}
            steer_strategies_results = {k: v for k, v in strategy_results.items() if k.startswith('steer_')}
            
            if amm_strategies:
                f.write("AMM Strategies:\n")
                for name, metrics in amm_strategies.items():
                    f.write(f"  {name:<20} APR: {metrics.get('apr', 0):6.2f}% "
                           f"MDD: {metrics.get('mdd', 0):6.2f}%\n")
                f.write("\n")
            
            if steer_strategies_results:
                f.write("Steer Strategies:\n")
                for name, metrics in steer_strategies_results.items():
                    f.write(f"  {name:<20} APR: {metrics.get('apr', 0):6.2f}% "
                           f"MDD: {metrics.get('mdd', 0):6.2f}%\n")
                f.write("\n")
            
            # 最佳策略推薦
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            best_apr = max(strategy_results.items(), key=lambda x: x[1].get('apr', 0))
            best_mdd = min(strategy_results.items(), key=lambda x: x[1].get('mdd', 100))
            best_sharpe = max(strategy_results.items(), key=lambda x: x[1].get('sharpe', 0))
            
            f.write(f"🏆 Best APR: {best_apr[0]} ({best_apr[1].get('apr', 0):.2f}%)\n")
            f.write(f"🛡️ Best MDD: {best_mdd[0]} ({best_mdd[1].get('mdd', 0):.2f}%)\n")
            f.write(f"📈 Best Sharpe: {best_sharpe[0]} ({best_sharpe[1].get('sharpe', 0):.2f})\n\n")
            
            # 策略建議
            f.write("STRATEGY RECOMMENDATIONS\n")
            f.write("-" * 25 + "\n")
            f.write("For High Returns: Use the strategy with highest APR\n")
            f.write("For Low Risk: Use the strategy with lowest MDD\n")
            f.write("For Balanced: Use the strategy with highest Sharpe ratio\n")
            f.write("For Innovation: Try Steer strategies for advanced features\n")
    
    print(f"📄 Integrated report saved to: {report_path}")

def main():
    """主函數"""
    
    # 配置
    pool = "BTCUSDC"
    frequency = "1d"
    study_name = f"integrated_{pool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    n_trials = 30
    
    try:
        results = run_integrated_backtest(
            pool=pool,
            frequency=frequency,
            study_name=study_name,
            n_trials=n_trials
        )
        
        print("\n🎉 Integration completed successfully!")
        print(f"📊 Total strategies tested: {len(results.get('strategy_results', {}))}")
        
    except Exception as e:
        print(f"❌ Error running integrated backtest: {e}")
        logger.error(f"Integrated backtest failed: {e}")

if __name__ == "__main__":
    main()
