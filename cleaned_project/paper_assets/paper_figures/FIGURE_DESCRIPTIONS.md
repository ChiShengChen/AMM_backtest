# 論文圖表詳細描述

本文檔詳細描述每張圖表的內容、用途和在論文中的建議位置。

## 📊 性能比較圖表 (performance_comparison/)

### 新增模型性能圖表
- `equity_curves_comparison.png` - 資金曲線比較圖
- `apr_comparison.png` - APR比較圖  
- `risk_return_scatter.png` - 風險收益散點圖
- `drawdown_analysis.png` - 回撤分析圖
- `performance_heatmap.png` - 性能熱力圖
- `comprehensive_dashboard.png` - 綜合儀表板

## 📊 原始性能比較圖表 (performance_comparison/)

### 1. unified_model_comparison.png
**描述**: 統一模型比較圖，包含8個子圖和一個排名表
**內容**:
- 子圖1: 各模型總體性能比較（Return, Sharpe Ratio, Max Drawdown）
- 子圖2: 按資產分類的性能比較（BTCUSDC, ETHUSDC, USDCUSDT）
- 子圖3: 按策略類型分類的性能比較（Classic ML, Quantum ML, QASA, PennyLane）
- 子圖4: 風險調整後收益比較
- 子圖5: 最大回撤分析
- 子圖6: 夏普比率分布
- 子圖7: 模型穩定性分析
- 子圖8: 綜合排名熱力圖
- 排名表: 54個模型的詳細排名和指標

**論文位置**: Results章節主要圖表
**用途**: 展示所有模型的綜合性能比較

### 2. performance_comparison.png
**描述**: 性能比較圖，展示不同模型類型的關鍵指標
**內容**:
- 各模型類型的平均收益比較
- 風險指標對比
- 性能分布箱線圖

**論文位置**: Results章節
**用途**: 支持量子vs經典性能對比論點

### 3. strategy_ranking.png
**描述**: 策略排名圖，按性能指標排序
**內容**:
- 前10名策略的詳細排名
- 各指標的相對表現
- 策略類型標識

**論文位置**: Results章節
**用途**: 突出最佳表現策略

### 4. quantum_vs_classic_detailed.png
**描述**: 量子vs經典詳細比較圖
**內容**:
- 量子模型和經典模型的直接對比
- 統計顯著性分析
- 性能差異的可視化

**論文位置**: Results章節核心圖表
**用途**: 證明量子模型的優勢

## 🏗️ 模型架構圖表 (model_architecture/)

### 5. model_architecture_analysis.png
**描述**: 模型架構分析圖
**內容**:
- 各模型類型的架構對比
- 參數數量比較
- 複雜度分析

**論文位置**: Methodology章節
**用途**: 說明實驗設計的模型多樣性

### 6. model_composition_analysis.png
**描述**: 模型組成分析圖
**內容**:
- 54個模型的詳細分類
- 各類別模型數量統計
- 特徵工程方法對比

**論文位置**: Methodology章節
**用途**: 展示實驗的全面性

## 🔍 特徵分析圖表 (feature_analysis/)

### 7. feature_importance_analysis.png
**描述**: 特徵重要性分析圖
**內容**:
- 122個經典特徵的重要性排序
- 前20個最重要特徵的詳細分析
- 特徵類別分布

**論文位置**: Methodology章節特徵工程部分
**用途**: 說明特徵選擇的科學性

### 8. angle_encoding_distribution.png
**描述**: 角度編碼分布圖
**內容**:
- 經典特徵到量子角度的映射分布
- 不同編碼方法的對比
- 角度範圍[0, 2π]的覆蓋情況

**論文位置**: Methodology章節量子特徵工程部分
**用途**: 展示角度編碼的有效性

### 9. feature_importance_vs_angles.png
**描述**: 特徵重要性vs角度映射圖
**內容**:
- 特徵重要性與對應角度的關係
- 映射質量的可視化
- 重要特徵的角度分布

**論文位置**: Methodology章節
**用途**: 驗證角度編碼的合理性

### 10. angle_correlation_matrix.png
**描述**: 角度相關性矩陣圖
**內容**:
- 量子角度特徵間的相關性
- 熱力圖顯示特徵關係
- 冗餘特徵識別

**論文位置**: Methodology章節
**用途**: 分析量子特徵的獨立性

## ⚛️ 量子分析圖表 (quantum_analysis/)

### 11. quantum_circuit_architecture.png
**描述**: 量子電路架構圖
**內容**:
- VQE Classifier電路設計
- QNN電路結構
- 量子門的排列和連接

**論文位置**: Methodology章節量子模型部分
**用途**: 說明量子電路的設計原理

### 12. angle_encoding_analysis.png
**描述**: 角度編碼分析圖
**內容**:
- 不同編碼方法的性能比較
- 編碼質量的量化指標
- 最佳編碼方法選擇

**論文位置**: Methodology章節
**用途**: 證明角度編碼方法的有效性

### 13. quantum_model_comparison.png
**描述**: 量子模型比較圖
**內容**:
- 不同量子模型的性能對比
- Qiskit vs PennyLane比較
- 量子模型內部差異分析

**論文位置**: Results章節
**用途**: 展示量子模型的多樣性

### 14. training_time_comparison.png
**描述**: 訓練時間比較圖
**內容**:
- 各模型類型的訓練時間對比
- 計算複雜度分析
- 效率vs性能的權衡

**論文位置**: Results章節或Discussion章節
**用途**: 討論量子模型的實用性

## 📈 回測結果圖表 (backtest_results/)

### 15. comprehensive_strategy_comparison.png
**描述**: AMM策略綜合比較圖
**內容**:
- 5年回測期間的權益曲線
- 各AMM策略的累積收益
- 風險調整後表現

**論文位置**: Results章節
**用途**: 展示AMM策略的長期表現

### 16. concentrated_liquidity_amm_comparison.png
**描述**: 集中流動性AMM比較圖
**內容**:
- 集中流動性策略的詳細分析
- 與傳統AMM的對比
- 流動性效率分析

**論文位置**: Results章節
**用途**: 說明不同AMM策略的差異

### 17. final_pennylane_quantum_backtest_summary.png
**描述**: PennyLane量子回測總結圖
**內容**:
- PennyLane量子模型的回測結果
- 性能指標總結
- 與基準的比較

**論文位置**: Results章節
**用途**: 展示純量子模型的表現

### 18. simplified_qasa_benchmark_summary.png
**描述**: QASA基準測試總結圖
**內容**:
- 量子-經典混合模型的結果
- 混合策略的優勢分析
- 基準測試表現

**論文位置**: Results章節
**用途**: 證明混合策略的有效性

## 📝 圖表使用建議

### 論文章節對應
- **Abstract**: 無需圖表
- **Introduction**: 模型架構圖表 (5, 6)
- **Literature Review**: 無需圖表
- **Methodology**: 特徵分析圖表 (7, 8, 9, 10, 11, 12)
- **Results**: 性能比較圖表 (1, 2, 3, 4, 13, 14, 15, 16, 17, 18)
- **Analysis**: 詳細分析圖表 (1, 4, 13, 14)
- **Conclusion**: 綜合比較圖表 (1, 4)

### 圖表品質特點
- **解析度**: 所有圖表都是高解析度PNG格式
- **標籤**: 統一的英文標題和軸標籤
- **配色**: 學術論文標準配色方案
- **字體**: 清晰可讀的字體大小
- **布局**: 專業的圖表布局和間距

### 引用建議
- 每張圖表都應該有對應的圖表標題和說明
- 在正文中引用圖表時使用 "Figure X" 格式
- 圖表說明應該簡潔但完整地描述圖表內容
- 重要發現應該在正文中詳細討論

## 📊 新增模型性能圖表詳細描述

### 19. equity_curves_comparison.png
**描述**: 資金曲線比較圖，展示各模型類型的累積收益曲線
**內容**:
- 4個子圖分別展示不同模型類型的資金曲線
- 經典ML、量子ML、QASA混合、PennyLane模型的對比
- 時間序列的累積收益變化
- 模型穩定性和趨勢分析

**論文位置**: Results章節
**用途**: 展示各模型類型的長期表現和穩定性

### 20. apr_comparison.png
**描述**: APR比較圖，包含柱狀圖和箱線圖
**內容**:
- 左圖: 各模型APR的柱狀圖比較
- 右圖: 按模型類型分組的APR分布箱線圖
- 數值標籤顯示具體APR百分比
- 模型類型顏色編碼

**論文位置**: Results章節
**用途**: 直觀比較各模型的年化收益率

### 21. risk_return_scatter.png
**描述**: 風險收益散點圖，展示風險與收益的關係
**內容**:
- X軸: 波動率(風險)
- Y軸: 年化收益率
- 點的大小: 夏普比率
- 等夏普比率線作為參考
- 模型名稱標籤

**論文位置**: Results章節
**用途**: 分析風險調整後收益和模型效率

### 22. drawdown_analysis.png
**描述**: 回撤分析圖，包含最大回撤比較和分布分析
**內容**:
- 左圖: 各模型最大回撤的柱狀圖比較
- 右圖: 按模型類型分組的回撤分布箱線圖
- 數值標籤顯示具體回撤百分比
- 風險控制能力分析

**論文位置**: Results章節
**用途**: 評估各模型的風險控制能力

### 23. performance_heatmap.png
**描述**: 性能熱力圖，標準化指標的綜合比較
**內容**:
- 行: 性能指標(APR, 夏普比率, 最大回撤, 波動率, 勝率)
- 列: 各模型
- 顏色: 標準化後的分數(紅色=高, 藍色=低)
- 數值標籤顯示標準化分數

**論文位置**: Results章節
**用途**: 快速識別各模型在各指標上的相對表現

### 24. comprehensive_dashboard.png
**描述**: 綜合儀表板，包含7個子圖的完整性能分析
**內容**:
- 子圖1: APR比較
- 子圖2: 夏普比率比較
- 子圖3: 最大回撤比較
- 子圖4: 波動率比較
- 子圖5: 風險收益散點圖(跨2列)
- 子圖6: 勝率比較
- 子圖7: 交易次數比較

**論文位置**: Results章節主要圖表
**用途**: 提供模型性能的全面視圖

## 📊 數據文件

### integrated_model_performance.csv
**描述**: 整合的模型性能數據
**內容**:
- 8個模型的完整性能指標
- 包含APR、夏普比率、最大回撤、波動率等
- 模型類型分類和數據來源標記

### integrated_model_performance.json
**描述**: JSON格式的性能數據
**內容**: 與CSV文件相同的數據，JSON格式便於程序處理

### performance_summary_report.md
**描述**: 性能摘要報告
**內容**:
- 基本統計信息
- 模型類型分布
- 性能摘要表格
- 最佳表現模型
- 圖表說明
