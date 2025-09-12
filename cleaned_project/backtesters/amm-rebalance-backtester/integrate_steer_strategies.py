#!/usr/bin/env python3
"""
整合 Steer Intent Backtester 策略到 AMM 回測系統
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# 添加steer_intent_backtester路徑
sys.path.append('/Users/michael/Desktop/Omnis_bt/steer_intent_backtester')

# 導入steer策略
from steerbt.strategies import (
    ClassicStrategy,
    ChannelMultiplierStrategy,
    BollingerStrategy,
    KeltnerStrategy,
    DonchianStrategy,
    StableStrategy,
    FluidStrategy,
    ImperfectClassicStrategy
)

# 導入AMM系統
from src.strategies.base import BaseStrategy
from src.core.engine import BacktestEngine
from src.io.loader import DataLoader
from src.opt.search import OptunaOptimizer
from src.reporting.plots import PlotGenerator
from src.reporting.tables import TableGenerator

logger = logging.getLogger(__name__)

class SteerStrategyAdapter(BaseStrategy):
    """適配器類，將Steer策略適配到AMM系統"""
    
    def __init__(self, steer_strategy, **kwargs):
        super().__init__(**kwargs)
        self.steer_strategy = steer_strategy
        self.name = f"Steer_{steer_strategy.__class__.__name__}"
        
    def calculate_ranges(
        self,
        price_data: pd.DataFrame,
        current_price: float,
        portfolio_value: float,
        **kwargs
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """適配Steer策略的calculate_range方法"""
        try:
            # 轉換數據格式以符合Steer策略要求
            steer_price_data = self._convert_price_data(price_data)
            
            # 調用Steer策略
            ranges, liquidities = self.steer_strategy.calculate_range(
                steer_price_data, current_price, portfolio_value
            )
            
            return ranges, liquidities
            
        except Exception as e:
            logger.error(f"Error in Steer strategy {self.name}: {e}")
            # 返回默認範圍
            width = current_price * 0.1  # 10% 寬度
            return [(current_price - width/2, current_price + width/2)], [portfolio_value]
    
    def should_rebalance(
        self,
        current_price: float,
        current_time: datetime,
        **kwargs
    ) -> bool:
        """適配再平衡邏輯"""
        # 簡化的再平衡邏輯
        if self.last_rebalance is None:
            return True
            
        time_diff = current_time - self.last_rebalance
        return time_diff.total_seconds() > 3600  # 1小時後再平衡
    
    def _convert_price_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """轉換價格數據格式以符合Steer策略要求"""
        # Steer策略期望的數據格式
        converted = pd.DataFrame()
        converted['timestamp'] = price_data.index
        converted['open'] = price_data['open']
        converted['high'] = price_data['high']
        converted['low'] = price_data['low']
        converted['close'] = price_data['close']
        converted['volume'] = price_data['volume']
        return converted

class IntegratedBacktester:
    """整合回測器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.steer_strategies = self._initialize_steer_strategies()
        
    def _initialize_steer_strategies(self) -> Dict[str, SteerStrategyAdapter]:
        """初始化Steer策略"""
        strategies = {}
        
        # 經典策略
        strategies['steer_classic'] = SteerStrategyAdapter(
            ClassicStrategy(
                width_mode='percentage',
                width_value=0.1,  # 10%
                trigger_mode='center_deviation',
                trigger_value=0.05,  # 5%
                curve_type='linear'
            )
        )
        
        # 通道倍數策略
        strategies['steer_channel_multiplier'] = SteerStrategyAdapter(
            ChannelMultiplierStrategy(
                width_pct=0.15,  # 15%
                trigger_pct=0.08  # 8%
            )
        )
        
        # 布林帶策略
        strategies['steer_bollinger'] = SteerStrategyAdapter(
            BollingerStrategy(
                period=20,
                std_multiplier=2.0,
                trigger_pct=0.1
            )
        )
        
        # 肯特納通道策略
        strategies['steer_keltner'] = SteerStrategyAdapter(
            KeltnerStrategy(
                ema_period=20,
                atr_multiplier=2.0,
                trigger_pct=0.1
            )
        )
        
        # 唐奇安通道策略
        strategies['steer_donchian'] = SteerStrategyAdapter(
            DonchianStrategy(
                period=20,
                width_multiplier=1.0,
                trigger_pct=0.1
            )
        )
        
        # 穩定策略
        strategies['steer_stable'] = SteerStrategyAdapter(
            StableStrategy(
                anchor_period=50,
                width_pct=0.2,
                num_bins=5,
                curve_type='gaussian'
            )
        )
        
        # 流體策略
        strategies['steer_fluid'] = SteerStrategyAdapter(
            FluidStrategy(
                ideal_ratio=0.5,
                imbalance_threshold=0.1,
                rebalance_threshold=0.05
            )
        )
        
        # 不完美經典策略
        strategies['steer_imperfect_classic'] = SteerStrategyAdapter(
            ImperfectClassicStrategy(
                width_mode='percentage',
                width_value=0.12,
                trigger_mode='center_deviation',
                trigger_value=0.06,
                imperfection_factor=0.1
            )
        )
        
        return strategies
    
    def run_integrated_backtest(
        self,
        pool: str,
        frequency: str,
        study_name: str,
        n_trials: int = 50
    ):
        """運行整合回測"""
        
        # 加載數據
        loader = DataLoader(self.config_path)
        price_data = loader.load_price_data(pool, frequency)
        
        # 初始化引擎
        engine = BacktestEngine(self.config_path)
        
        # 添加Steer策略到引擎
        for name, strategy in self.steer_strategies.items():
            engine.add_strategy(name, strategy)
        
        # 運行優化
        optimizer = OptunaOptimizer(engine)
        best_params = optimizer.optimize(
            study_name=study_name,
            n_trials=n_trials
        )
        
        # 運行最終評估
        results = engine.run_full_evaluation(best_params)
        
        # 生成報告
        self._generate_reports(results, pool, study_name)
        
        return results
    
    def _generate_reports(self, results: Dict, pool: str, study_name: str):
        """生成整合報告"""
        
        # 生成圖表
        plot_generator = PlotGenerator()
        plot_generator.generate_all_plots(results, pool)
        
        # 生成表格
        table_generator = TableGenerator()
        table_generator.generate_summary_tables(results)
        
        # 生成整合報告
        self._generate_integrated_report(results, pool, study_name)
    
    def _generate_integrated_report(self, results: Dict, pool: str, study_name: str):
        """生成整合策略報告"""
        
        report_path = f"results/integrated_strategy_report_{study_name}.txt"
        
        with open(report_path, 'w') as f:
            f.write("Integrated Strategy Backtest Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Pool: {pool}\n")
            f.write(f"Study: {study_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("STRATEGY COMPARISON\n")
            f.write("-" * 30 + "\n")
            
            # 原始AMM策略
            f.write("Original AMM Strategies:\n")
            for strategy_name, metrics in results.get('amm_strategies', {}).items():
                f.write(f"  {strategy_name}:\n")
                f.write(f"    APR: {metrics.get('apr', 0):.2f}%\n")
                f.write(f"    MDD: {metrics.get('mdd', 0):.2f}%\n")
                f.write(f"    Sharpe: {metrics.get('sharpe', 0):.2f}\n")
                f.write(f"    Rebalances: {metrics.get('rebalances', 0)}\n\n")
            
            # Steer策略
            f.write("Steer Intent Strategies:\n")
            for strategy_name, metrics in results.get('steer_strategies', {}).items():
                f.write(f"  {strategy_name}:\n")
                f.write(f"    APR: {metrics.get('apr', 0):.2f}%\n")
                f.write(f"    MDD: {metrics.get('mdd', 0):.2f}%\n")
                f.write(f"    Sharpe: {metrics.get('sharpe', 0):.2f}\n")
                f.write(f"    Rebalances: {metrics.get('rebalances', 0)}\n\n")
            
            # 最佳策略推薦
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            
            all_strategies = {}
            all_strategies.update(results.get('amm_strategies', {}))
            all_strategies.update(results.get('steer_strategies', {}))
            
            if all_strategies:
                best_apr = max(all_strategies.items(), key=lambda x: x[1].get('apr', 0))
                best_mdd = min(all_strategies.items(), key=lambda x: x[1].get('mdd', 100))
                best_sharpe = max(all_strategies.items(), key=lambda x: x[1].get('sharpe', 0))
                
                f.write(f"Best APR: {best_apr[0]} ({best_apr[1].get('apr', 0):.2f}%)\n")
                f.write(f"Best MDD: {best_mdd[0]} ({best_mdd[1].get('mdd', 0):.2f}%)\n")
                f.write(f"Best Sharpe: {best_sharpe[0]} ({best_sharpe[1].get('sharpe', 0):.2f})\n")
        
        print(f"Integrated report saved to: {report_path}")

def main():
    """主函數"""
    
    # 配置
    config_path = "configs/btcusdc_experiment.yaml"
    pool = "BTCUSDC"
    frequency = "1d"
    study_name = f"integrated_{pool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    n_trials = 30
    
    # 創建整合回測器
    backtester = IntegratedBacktester(config_path)
    
    # 運行整合回測
    print(f"Running integrated backtest for {pool}...")
    print(f"Study: {study_name}")
    print(f"Trials: {n_trials}")
    print(f"Strategies: {len(backtester.steer_strategies)} Steer strategies + Original AMM strategies")
    
    try:
        results = backtester.run_integrated_backtest(
            pool=pool,
            frequency=frequency,
            study_name=study_name,
            n_trials=n_trials
        )
        
        print("Integrated backtest completed successfully!")
        print(f"Results saved to results/ and reports/figs/{pool}/")
        
    except Exception as e:
        print(f"Error running integrated backtest: {e}")
        logger.error(f"Integrated backtest failed: {e}")

if __name__ == "__main__":
    main()
