#!/usr/bin/env python3
"""
詳細測試修正後的steer回測，包括現金耗盡的邊界情況
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加路徑
sys.path.append('/Users/michael/Desktop/Omnis_bt/steer_intent_backtester')
sys.path.append('/Users/michael/Desktop/Omnis_bt/cleaned_project/backtesters/steer_intent_backtester')

from steerbt.backtester import Backtester
from steerbt.strategies.classic import ClassicStrategy
from steerbt.portfolio import Portfolio

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_stress_test_data(n_points=20000):
    """創建壓力測試數據，包含劇烈波動"""
    logger.info(f"🔧 創建 {n_points} 個數據點的壓力測試數據...")
    
    # 創建模擬的ETH價格數據
    base_price = 2000.0
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='h')
    
    # 生成價格數據（帶有劇烈波動）
    np.random.seed(42)
    returns = np.random.normal(0, 0.05, n_points)  # 5% 標準差，更劇烈
    
    # 添加多個劇烈波動事件
    for i in range(100, n_points, 500):
        if i < n_points:
            # 20% 下跌
            returns[i:i+20] -= 0.2
        if i + 250 < n_points:
            # 15% 上漲
            returns[i+250:i+270] += 0.15
    
    # 添加一些極端事件
    for i in range(1000, n_points, 2000):
        if i < n_points:
            # 50% 暴跌
            returns[i:i+50] -= 0.5
        if i + 1000 < n_points:
            # 30% 暴漲
            returns[i+1000:i+1020] += 0.3
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_points),
        'quote_volume': np.random.uniform(1000000, 10000000, n_points)
    }, index=dates)
    
    return data

def test_cash_management():
    """測試現金管理功能"""
    logger.info("🧪 測試現金管理功能...")
    
    # 創建壓力測試數據
    data = create_stress_test_data(20000)
    logger.info(f"📊 數據範圍: {data.index[0]} 到 {data.index[-1]}")
    logger.info(f"💰 價格範圍: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 測試不同的配置
    test_configs = [
        {
            "name": "保守配置",
            "liquidity_scale": 0.001,
            "fee_bps": 5,
            "width_value": 10.0
        },
        {
            "name": "中等配置", 
            "liquidity_scale": 0.01,
            "fee_bps": 10,
            "width_value": 5.0
        },
        {
            "name": "激進配置",
            "liquidity_scale": 0.05,
            "fee_bps": 20,
            "width_value": 2.0
        }
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n🔧 測試 {config['name']}...")
        
        # 創建回測配置
        backtest_config = {
            "pair": "ETHUSDC",
            "interval": "1h",
            "strategy": "classic",
            "strategy_params": {
                "width_mode": "percent",
                "width_value": config["width_value"],
                "placement_mode": "center",
                "curve_type": "uniform",
                "liquidity_scale": config["liquidity_scale"]
            },
            "initial_cash": 10000.0,
            "fee_bps": config["fee_bps"],
            "slippage_bps": 1,
            "gas_cost": 0.0,
            "liq_share": 0.001,
            "start_date": data.index[0],
            "end_date": data.index[-1]
        }
        
        try:
            # 創建回測器
            backtester = Backtester(backtest_config)
            
            # 運行回測
            results_data = backtester.run(data)
            
            if results_data is None:
                logger.error(f"❌ {config['name']} 回測失敗")
                continue
            
            # 獲取投資組合
            portfolio = backtester.portfolio
            
            # 檢查結果
            final_cash = portfolio.cash
            final_value = portfolio.get_total_value(data['close'].iloc[-1])
            rebalance_count = len(portfolio.transaction_history)
            total_fees = portfolio.total_fees_paid
            
            # 計算回報率
            initial_cash = backtest_config["initial_cash"]
            total_return = (final_value - initial_cash) / initial_cash * 100
            
            result = {
                "name": config['name'],
                "final_cash": final_cash,
                "final_value": final_value,
                "total_return": total_return,
                "rebalance_count": rebalance_count,
                "total_fees": total_fees,
                "cash_ratio": final_cash / final_value if final_value > 0 else 0,
                "success": final_value > initial_cash * 0.1  # 至少保留10%的價值
            }
            
            results.append(result)
            
            logger.info(f"💰 最終現金: ${final_cash:.2f}")
            logger.info(f"💎 最終總價值: ${final_value:.2f}")
            logger.info(f"📈 總回報率: {total_return:.2f}%")
            logger.info(f"🔄 重新平衡次數: {rebalance_count}")
            logger.info(f"💸 總手續費: ${total_fees:.2f}")
            logger.info(f"💵 現金比例: {result['cash_ratio']:.2%}")
            
            if result['success']:
                logger.info(f"✅ {config['name']} 測試通過")
            else:
                logger.warning(f"⚠️  {config['name']} 測試警告：總價值過低")
                
        except Exception as e:
            logger.error(f"❌ {config['name']} 測試失敗: {e}")
            continue
    
    # 總結結果
    logger.info("\n📊 測試結果總結:")
    logger.info("=" * 60)
    
    for result in results:
        status = "✅ 通過" if result['success'] else "❌ 失敗"
        logger.info(f"{result['name']:12} | 現金: ${result['final_cash']:8.2f} | 總價值: ${result['final_value']:8.2f} | 回報: {result['total_return']:6.2f}% | {status}")
    
    # 檢查是否有現金耗盡的情況
    cash_depleted = any(r['final_cash'] <= 0 for r in results)
    if cash_depleted:
        logger.warning("⚠️  發現現金耗盡的情況")
    else:
        logger.info("✅ 所有配置都保持了現金餘額")
    
    return results

def main():
    """主函數"""
    logger.info("🔧 詳細測試修正後的steer回測系統")
    
    try:
        results = test_cash_management()
        
        # 檢查整體成功率
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        
        if success_count == total_count:
            logger.info("🎉 所有測試都通過！steer回測修正成功")
        elif success_count > 0:
            logger.info(f"⚠️  {success_count}/{total_count} 測試通過，部分配置需要調整")
        else:
            logger.error("❌ 所有測試都失敗，需要進一步修正")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
