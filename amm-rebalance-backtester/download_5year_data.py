#!/usr/bin/env python3
"""
批量下載5年日線數據腳本
支持 ETH/USDC, BTC/USDC, USDT/USDC
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_5year_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceDataDownloader:
    """Binance 數據下載器"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AMMBacktester/1.0"
        })
        
        # 支持的交易對
        self.symbols = {
            "ETHUSDC": "ETHUSDC",
            "BTCUSDC": "BTCUSDC", 
            "USDCUSDT": "USDCUSDT"  # 修正：使用正確的 Binance 交易對名稱
        }
        
        # 時間間隔映射
        self.interval = "1d"
        
    def fetch_klines(self, symbol: str, start_time: int, end_time: int, limit: int = 1000) -> list:
        """獲取 K 線數據"""
        params = {
            "symbol": symbol,
            "interval": self.interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v3/klines",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"獲取 {symbol} 數據失敗: {e}")
            return []
    
    def download_symbol_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """下載指定交易對的完整數據"""
        logger.info(f"開始下載 {symbol} 從 {start_date.date()} 到 {end_date.date()}")
        
        # 轉換為毫秒時間戳
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        all_klines = []
        current_start = start_ts
        
        while current_start < end_ts:
            current_end = min(current_start + (1000 * 24 * 60 * 60 * 1000), end_ts)  # 1000天
            
            logger.info(f"下載 {symbol} 從 {datetime.fromtimestamp(current_start/1000).date()} 到 {datetime.fromtimestamp(current_end/1000).date()}")
            
            klines = self.fetch_klines(symbol, current_start, current_end, 1000)
            
            if not klines:
                logger.warning(f"沒有獲取到 {symbol} 的數據")
                break
            
            all_klines.extend(klines)
            
            # 更新開始時間
            current_start = current_end + 1
            
            # 速率限制：1200 requests/min，我們保守一點
            time.sleep(0.1)
        
        if not all_klines:
            logger.error(f"沒有獲取到 {symbol} 的任何數據")
            return pd.DataFrame()
        
        # 轉換為 DataFrame
        df = pd.DataFrame(all_klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # 轉換數據類型
        numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 轉換時間戳
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        # 排序並去重
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        
        logger.info(f"成功下載 {symbol}: {len(df)} 條記錄")
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str, output_dir: str):
        """保存數據到文件"""
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        start_date = df["timestamp"].min().strftime("%Y%m%d")
        end_date = df["timestamp"].max().strftime("%Y%m%d")
        filename = f"{symbol}_1d_{start_date}_{end_date}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # 保存到 CSV
        df.to_csv(filepath, index=False)
        logger.info(f"數據已保存到: {filepath}")
        
        # 顯示數據摘要
        logger.info(f"數據摘要:")
        logger.info(f"  時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        logger.info(f"  記錄數量: {len(df)}")
        logger.info(f"  價格範圍: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        logger.info(f"  總交易量: {df['volume'].sum():.2f}")
        
        return filepath

def main():
    """主函數"""
    import argparse
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='批量下載5年日線數據')
    parser.add_argument('--symbol', help='指定交易對 (可選，不指定則下載所有)')
    parser.add_argument('--start', help='開始日期 (YYYY-MM-DD，可選)')
    parser.add_argument('--end', help='結束日期 (YYYY-MM-DD，可選)')
    parser.add_argument('--out', help='輸出目錄 (可選)')
    
    args = parser.parse_args()
    
    # 創建下載器
    downloader = BinanceDataDownloader()
    
    # 設置時間範圍
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    else:
        end_date = datetime.now()
        
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=5*365)  # 5年前
    
    logger.info(f"下載時間範圍: {start_date.date()} 到 {end_date.date()}")
    
    # 創建數據目錄
    if args.out:
        data_dir = args.out
    else:
        data_dir = "data/5year_daily"
    os.makedirs(data_dir, exist_ok=True)
    
    # 確定要下載的交易對
    if args.symbol:
        # 下載指定交易對
        if args.symbol in downloader.symbols:
            symbols_to_download = {args.symbol: downloader.symbols[args.symbol]}
        else:
            logger.error(f"不支持的交易對: {args.symbol}")
            return
    else:
        # 下載所有交易對
        symbols_to_download = downloader.symbols
    
    # 下載數據
    success_count = 0
    total_count = len(symbols_to_download)
    
    for symbol_name, symbol_code in symbols_to_download.items():
        try:
            logger.info(f"正在處理 {symbol_name} ({symbol_code})")
            
            # 下載數據
            df = downloader.download_symbol_data(symbol_code, start_date, end_date)
            
            if not df.empty:
                # 保存數據
                filepath = downloader.save_data(df, symbol_code, data_dir)
                success_count += 1
                logger.info(f"✅ {symbol_name} 下載成功")
            else:
                logger.error(f"❌ {symbol_name} 下載失敗：沒有數據")
                
        except Exception as e:
            logger.error(f"❌ {symbol_name} 下載失敗: {e}")
        
        # 在交易對之間添加延遲
        if symbol_name != list(symbols_to_download.keys())[-1]:
            logger.info("等待 5 秒後繼續下一個交易對...")
            time.sleep(5)
    
    # 總結報告
    logger.info("="*60)
    logger.info("下載完成總結")
    logger.info("="*60)
    logger.info(f"總交易對數: {total_count}")
    logger.info(f"成功下載: {success_count}")
    logger.info(f"失敗數量: {total_count - success_count}")
    logger.info(f"數據保存目錄: {data_dir}")
    
    if success_count == total_count:
        logger.info("🎉 所有交易對數據下載成功！")
    else:
        logger.warning("⚠️ 部分交易對數據下載失敗，請檢查日誌")
    
    # 顯示下載的文件
    logger.info("\n下載的文件:")
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            filepath = os.path.join(data_dir, file)
            size = os.path.getsize(filepath) / 1024  # KB
            logger.info(f"  {file} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
