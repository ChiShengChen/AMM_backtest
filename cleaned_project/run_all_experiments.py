#!/usr/bin/env python3
"""
統一執行腳本 - 運行所有核心實驗
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path, description):
    """運行腳本並記錄結果"""
    logger.info(f"🚀 開始執行: {description}")
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            logger.info(f"✅ {description} 完成")
            return True
        else:
            logger.error(f"❌ {description} 失敗: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ {description} 執行錯誤: {e}")
        return False

def main():
    """主執行函數"""
    logger.info("🎯 開始執行所有核心實驗...")
    
    # 確保在正確的目錄
    os.chdir(Path(__file__).parent)
    
    experiments = [
        {
            "script": "core_scripts/improved_training_demo.py",
            "description": "經典機器學習訓練演示"
        },
        {
            "script": "core_scripts/improved_quantum_training_demo.py", 
            "description": "量子機器學習訓練演示"
        },
        {
            "script": "core_scripts/simplified_qasa_benchmark.py",
            "description": "QASA基準測試"
        },
        {
            "script": "analysis_tools/create_unified_model_comparison.py",
            "description": "統一模型比較分析"
        },
        {
            "script": "analysis_tools/analyze_classical_features_for_quantum.py",
            "description": "經典特徵分析"
        },
        {
            "script": "analysis_tools/generate_all_performance_charts.py",
            "description": "生成模型性能比較圖表"
        }
    ]
    
    success_count = 0
    total_count = len(experiments)
    
    for exp in experiments:
        if run_script(exp["script"], exp["description"]):
            success_count += 1
    
    logger.info(f"📊 實驗完成: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        logger.info("🎉 所有實驗執行成功！")
        logger.info("📁 結果位置:")
        logger.info("  - 訓練結果: reports/")
        logger.info("  - 論文圖表: paper_assets/paper_figures/")
    else:
        logger.warning("⚠️  部分實驗失敗，請檢查日誌")

if __name__ == "__main__":
    main()
