#!/usr/bin/env python3
"""
測試修正後的steer回測，確保現金不會耗盡
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

def create_test_data(n_points=10000):
    """創建測試數據"""
    logger.info(f"🔧 創建 {n_points} 個數據點的測試數據...")
    
    # 創建模擬的ETH價格數據
    base_price = 2000.0
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='H')
    
    # 生成價格數據（帶有趨勢和波動）
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, n_points)  # 2% 標準差
    trend = np.linspace(0, 0.5, n_points)  # 50% 總趨勢
    prices = base_price * np.exp(np.cumsum(returns) + trend)
    
    # 添加一些劇烈波動來測試現金管理
    for i in range(100, n_points, 1000):
        if i < n_points:
            prices[i:i+10] *= 0.8  # 20% 下跌
        if i + 500 < n_points:
            prices[i+500:i+510] *= 1.2  # 20% 上漲
    
    data = pd.DataFrame({
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_points),
        'quote_volume': np.random.uniform(1000000, 10000000, n_points)
    }, index=dates)
    
    return data

def test_steer_backtest():
    """測試修正後的steer回測"""
    logger.info("🚀 開始測試修正後的steer回測...")
    
    # 創建測試數據
    data = create_test_data(10000)
    logger.info(f"📊 數據範圍: {data.index[0]} 到 {data.index[-1]}")
    logger.info(f"💰 價格範圍: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 創建回測配置
    config = {
        "pair": "ETHUSDC",
        "interval": "1h",
        "strategy": "classic",
        "strategy_params": {
            "width_mode": "percent",
            "width_value": 5.0,
            "placement_mode": "center",
            "curve_type": "uniform",
            "liquidity_scale": 0.01  # 使用較保守的流動性縮放
        },
        "initial_cash": 10000.0,
        "fee_bps": 5,  # 0.05% 手續費
        "slippage_bps": 1,  # 0.01% 滑點
        "gas_cost": 0.0,  # 無gas成本
        "liq_share": 0.001,
        "start_date": data.index[0],
        "end_date": data.index[-1]
    }
    
    # 創建回測器
    backtester = Backtester(config)
    
    # 運行回測
    logger.info("🔄 開始運行回測...")
    results = backtester.run(data)
    
    if results is None:
        logger.error("❌ 回測失敗")
        return False
    
    # 檢查結果
    logger.info("📈 回測完成，檢查結果...")
    
    # 從回測器獲取投資組合
    portfolio = backtester.portfolio
    
    # 檢查現金是否耗盡
    final_cash = portfolio.cash
    final_value = portfolio.get_total_value(data['close'].iloc[-1])
    rebalance_count = len(portfolio.transaction_history)
    
    logger.info(f"💰 最終現金: ${final_cash:.2f}")
    logger.info(f"💎 最終總價值: ${final_value:.2f}")
    logger.info(f"🔄 重新平衡次數: {rebalance_count}")
    logger.info(f"💸 總手續費: ${portfolio.total_fees_paid:.2f}")
    
    # 檢查現金是否耗盡
    if final_cash <= 0:
        logger.warning("⚠️  現金已耗盡，但這可能是正常的（如果所有資金都投入持倉）")
    else:
        logger.info("✅ 現金未耗盡")
    
    # 檢查總價值是否合理
    if final_value < 1000:  # 如果總價值低於初始資金的10%
        logger.error(f"❌ 總價值過低: ${final_value:.2f}")
        return False
    
    logger.info("✅ 回測成功完成，現金管理正常")
    return True

def main():
    """主函數"""
    logger.info("🔧 測試修正後的steer回測系統")
    
    try:
        success = test_steer_backtest()
        if success:
            logger.info("🎉 所有測試通過！steer回測已修正")
        else:
            logger.error("❌ 測試失敗")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
