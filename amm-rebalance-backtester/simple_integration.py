#!/usr/bin/env python3
"""
簡單的整合測試 - 直接比較AMM和Steer策略
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

logger = logging.getLogger(__name__)

def test_steer_strategies():
    """測試Steer策略的基本功能"""
    
    print("🧪 Testing Steer Intent Backtester Strategies")
    print("=" * 50)
    
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
        
        print("✅ Successfully imported Steer strategies")
        
        # 創建測試數據
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 100 + np.random.randn(100).cumsum() + 2,
            'low': 100 + np.random.randn(100).cumsum() - 2,
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        print(f"📊 Created test data: {len(test_data)} points")
        
        # 測試策略
        strategies_to_test = [
            ("Classic", ClassicStrategy, {
                'width_mode': 'percentage',
                'width_value': 0.1,
                'trigger_mode': 'center_deviation',
                'trigger_value': 0.05,
                'curve_type': 'linear',
                'placement_mode': 'center'
            }),
            ("Channel Multiplier", ChannelMultiplierStrategy, {
                'width_pct': 0.15,
                'trigger_pct': 0.08
            }),
            ("Bollinger", BollingerStrategy, {
                'n': 20,
                'k': 2.0,
                'trigger_pct': 0.1
            }),
            ("Keltner", KeltnerStrategy, {
                'n': 20,
                'm': 2.0,
                'trigger_pct': 0.1
            }),
            ("Donchian", DonchianStrategy, {
                'n': 20,
                'trigger_pct': 0.1
            }),
            ("Stable", StableStrategy, {
                'peg_method': 'sma',
                'anchor_period': 50,
                'width_pct': 0.2,
                'num_bins': 5,
                'curve_type': 'gaussian'
            }),
            ("Fluid", FluidStrategy, {
                'acceptable_ratio': 0.5,
                'imbalance_threshold': 0.1,
                'rebalance_threshold': 0.05
            })
        ]
        
        results = {}
        
        for name, strategy_class, params in strategies_to_test:
            try:
                print(f"\n🔧 Testing {name} strategy...")
                
                # 創建策略實例
                strategy = strategy_class(**params)
                
                # 測試計算範圍
                current_price = 100.0
                portfolio_value = 10000.0
                
                ranges, liquidities = strategy.calculate_range(
                    test_data, current_price, portfolio_value
                )
                
                print(f"  ✅ {name}: Generated {len(ranges)} ranges")
                print(f"     Price range: {ranges[0] if ranges else 'N/A'}")
                print(f"     Liquidity: {liquidities[0] if liquidities else 'N/A'}")
                
                results[name] = {
                    'status': 'success',
                    'ranges': len(ranges),
                    'price_range': ranges[0] if ranges else None,
                    'liquidity': liquidities[0] if liquidities else None
                }
                
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                results[name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # 總結結果
        print("\n📋 Test Results Summary")
        print("-" * 30)
        
        successful = [name for name, result in results.items() if result['status'] == 'success']
        failed = [name for name, result in results.items() if result['status'] == 'failed']
        
        print(f"✅ Successful: {len(successful)} strategies")
        for name in successful:
            print(f"   - {name}")
        
        print(f"❌ Failed: {len(failed)} strategies")
        for name in failed:
            print(f"   - {name}: {results[name]['error']}")
        
        return results
        
    except ImportError as e:
        print(f"❌ Failed to import Steer strategies: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def compare_with_amm_strategies():
    """比較Steer策略與AMM策略"""
    
    print("\n🔄 Comparing with AMM Strategies")
    print("=" * 40)
    
    try:
        # 導入AMM策略
        from src.strategies import (
            BaselineStaticStrategy,
            BaselineFixedStrategy,
            DynamicVolatilityStrategy,
            DynamicInventoryStrategy
        )
        
        print("✅ Successfully imported AMM strategies")
        
        # 創建測試數據
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 100 + np.random.randn(100).cumsum() + 2,
            'low': 100 + np.random.randn(100).cumsum() - 2,
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # 測試AMM策略
        amm_strategies = [
            ("Baseline Static", BaselineStaticStrategy, {
                'width_pct': 500.0,
                'rebalance_cooldown_hours': 168
            }),
            ("Baseline Fixed", BaselineFixedStrategy, {
                'width_pct': 50.0,
                'price_deviation_bps': 50,
                'rebalance_cooldown_hours': 24
            }),
            ("Dynamic Vol", DynamicVolatilityStrategy, {
                'vol_estimator': 'ewma',
                'k_width': 1.5,
                'price_deviation_bps': 50,
                'rebalance_cooldown_hours': 24
            }),
            ("Dynamic Inventory", DynamicInventoryStrategy, {
                'skew_threshold_pct': 15.0,
                'fee_density_window_h': 24,
                'reinvest_frequency_h': 48
            })
        ]
        
        amm_results = {}
        
        for name, strategy_class, params in amm_strategies:
            try:
                print(f"\n🔧 Testing AMM {name} strategy...")
                
                strategy = strategy_class(**params)
                
                current_price = 100.0
                portfolio_value = 10000.0
                
                ranges, liquidities = strategy.calculate_ranges(
                    test_data, current_price, portfolio_value
                )
                
                print(f"  ✅ {name}: Generated {len(ranges)} ranges")
                print(f"     Price range: {ranges[0] if ranges else 'N/A'}")
                print(f"     Liquidity: {liquidities[0] if liquidities else 'N/A'}")
                
                amm_results[name] = {
                    'status': 'success',
                    'ranges': len(ranges),
                    'price_range': ranges[0] if ranges else None,
                    'liquidity': liquidities[0] if liquidities else None
                }
                
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                amm_results[name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return amm_results
        
    except ImportError as e:
        print(f"❌ Failed to import AMM strategies: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def generate_comparison_report(steer_results, amm_results):
    """生成比較報告"""
    
    if not steer_results or not amm_results:
        print("❌ Cannot generate report - missing results")
        return
    
    report_path = f"results/strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w') as f:
        f.write("Strategy Comparison Report\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("STEER INTENT STRATEGIES\n")
        f.write("-" * 25 + "\n")
        
        for name, result in steer_results.items():
            if result['status'] == 'success':
                f.write(f"✅ {name}:\n")
                f.write(f"   Ranges: {result['ranges']}\n")
                f.write(f"   Price Range: {result['price_range']}\n")
                f.write(f"   Liquidity: {result['liquidity']}\n\n")
            else:
                f.write(f"❌ {name}: {result['error']}\n\n")
        
        f.write("AMM STRATEGIES\n")
        f.write("-" * 15 + "\n")
        
        for name, result in amm_results.items():
            if result['status'] == 'success':
                f.write(f"✅ {name}:\n")
                f.write(f"   Ranges: {result['ranges']}\n")
                f.write(f"   Price Range: {result['price_range']}\n")
                f.write(f"   Liquidity: {result['liquidity']}\n\n")
            else:
                f.write(f"❌ {name}: {result['error']}\n\n")
        
        # 統計
        steer_success = sum(1 for r in steer_results.values() if r['status'] == 'success')
        amm_success = sum(1 for r in amm_results.values() if r['status'] == 'success')
        
        f.write("SUMMARY\n")
        f.write("-" * 10 + "\n")
        f.write(f"Steer Strategies: {steer_success}/{len(steer_results)} successful\n")
        f.write(f"AMM Strategies: {amm_success}/{len(amm_results)} successful\n")
        f.write(f"Total Strategies: {steer_success + amm_success}/{len(steer_results) + len(amm_results)} successful\n")
    
    print(f"📄 Comparison report saved to: {report_path}")

def main():
    """主函數"""
    
    print("🚀 Starting Strategy Integration Test")
    print("=" * 50)
    
    # 測試Steer策略
    steer_results = test_steer_strategies()
    
    # 測試AMM策略
    amm_results = compare_with_amm_strategies()
    
    # 生成比較報告
    if steer_results and amm_results:
        generate_comparison_report(steer_results, amm_results)
    
    print("\n🎉 Integration test completed!")
    print("Check the results/ directory for detailed reports.")

if __name__ == "__main__":
    main()
