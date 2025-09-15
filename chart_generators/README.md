# Chart Generators - 圖表生成器

這個資料夾包含了所有用於生成steer策略比較圖表的Python腳本。

## 📊 腳本分類

### 🔧 核心腳本

#### 1. **run_steer_comparison.py**
- **用途**: 最初的steer策略修復和比較
- **功能**: 
  - 修復steer策略的現金耗盡問題
  - 生成基礎的steer策略比較圖表
  - 輸出重新平衡次數比較表格
- **輸出**: `steer_comparison_results/` 目錄

#### 2. **create_enhanced_comparison_charts.py**
- **用途**: 增強版比較圖表生成器
- **功能**:
  - 生成類似QML/ML風格的比較圖表
  - 包含準確率比較、重新平衡次數比較、權益曲線等
- **輸出**: `simplified_ultimate_comparison/` 目錄

#### 3. **create_improved_comparison_charts.py**
- **用途**: 改進版圖表生成器
- **功能**:
  - 在效率分析圖中添加模型名稱標籤
  - 移除性能熱力圖的標準化處理
  - 按重新平衡次數從大到小排序
- **輸出**: 更新 `simplified_ultimate_comparison/` 目錄

#### 4. **create_enhanced_efficiency_analysis.py**
- **用途**: 最終版效率分析生成器
- **功能**:
  - 在風險回報圖中添加夏普比率參考線 (0.5, 1.0, 1.5, 2.0)
  - 生成最完整的效率分析圖表
- **輸出**: 更新 `simplified_ultimate_comparison/` 目錄

### 📈 擴展腳本

#### 5. **create_steer_ml_comparison.py**
- **用途**: Steer策略與ML/QML模型比較
- **功能**: 生成steer策略與機器學習模型的對比圖表

#### 6. **create_detailed_steer_ml_comparison.py**
- **用途**: 詳細的Steer vs ML比較
- **功能**: 生成更詳細的比較分析，包括APR、性能曲線等

#### 7. **create_comprehensive_steer_comparison.py**
- **用途**: 全面的Steer策略比較
- **功能**: 分析所有16種steer策略的綜合性能

#### 8. **create_simplified_ultimate_comparison.py**
- **用途**: 簡化版終極比較
- **功能**: 整合所有比較結果，生成最終排名和報告

### 📋 報告生成腳本

#### 9. **create_final_enhanced_report.py**
- **用途**: 最終增強版報告生成器
- **功能**: 生成包含所有圖表的完整分析報告

#### 10. **create_final_summary_report.py**
- **用途**: 最終總結報告生成器
- **功能**: 生成簡化的總結報告

#### 11. **create_summary_visualization.py**
- **用途**: 總結視覺化生成器
- **功能**: 生成簡化的視覺化圖表

### 🧪 測試腳本

#### 12. **test_steer_fix.py**
- **用途**: Steer策略修復測試
- **功能**: 驗證steer策略修復後的基本功能

#### 13. **test_steer_detailed.py**
- **用途**: 詳細的Steer策略測試
- **功能**: 全面測試steer策略的各種配置

## 🚀 使用指南

### 基本使用流程

1. **修復和基礎比較**:
   ```bash
   python run_steer_comparison.py
   ```

2. **生成增強版圖表**:
   ```bash
   python create_enhanced_comparison_charts.py
   ```

3. **生成改進版圖表**:
   ```bash
   python create_improved_comparison_charts.py
   ```

4. **生成最終版效率分析**:
   ```bash
   python create_enhanced_efficiency_analysis.py
   ```

### 完整分析流程

```bash
# 1. 基礎分析
python run_steer_comparison.py

# 2. 增強版比較
python create_enhanced_comparison_charts.py

# 3. 改進版圖表
python create_improved_comparison_charts.py

# 4. 最終版效率分析
python create_enhanced_efficiency_analysis.py

# 5. 生成最終報告
python create_final_enhanced_report.py
```

## 📁 輸出目錄

- `steer_comparison_results/` - 基礎steer策略比較結果
- `simplified_ultimate_comparison/` - 終極比較結果
- `steer_ml_comparison_results/` - Steer vs ML比較結果
- `detailed_steer_ml_comparison/` - 詳細Steer vs ML比較結果
- `comprehensive_steer_comparison/` - 全面Steer策略比較結果

## 🎯 主要圖表類型

1. **準確率比較圖** - 條形圖 + 箱線圖
2. **重新平衡次數比較圖** - 條形圖 + 箱線圖
3. **權益曲線與不確定性帶** - 時間序列圖
4. **性能熱力圖** - 標準化/非標準化熱力圖
5. **效率分析圖** - 四面板散點圖分析
6. **策略排名表** - 綜合性能排名

## 📊 圖表特色

- **英文標題和標籤** - 所有圖表都使用英文
- **顏色編碼** - 按策略類型進行顏色區分
- **高解析度** - 300 DPI輸出品質
- **專業格式** - 符合學術和商業報告標準
- **參考線** - 包含夏普比率等關鍵指標參考線

## 🔧 依賴套件

- pandas
- numpy
- matplotlib
- seaborn
- pathlib
- logging
- warnings

## 📝 注意事項

1. 確保在運行腳本前已安裝所有依賴套件
2. 某些腳本需要先運行前置腳本才能正常工作
3. 輸出目錄會自動創建
4. 所有圖表都保存在對應的結果目錄中
