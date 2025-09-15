# Chart Generators - Steer Strategy Analysis

This directory contains all Python scripts for generating comprehensive steer strategy comparison charts and analysis reports.

## 📊 Script Categories

### 🔧 Core Scripts

#### 1. **run_steer_comparison.py**
- **Purpose**: Initial steer strategy fix and comparison
- **Features**: 
  - Fix steer strategy cash depletion issues
  - Generate basic steer strategy comparison charts
  - Output rebalance count comparison tables
- **Output**: `steer_comparison_results/` directory

#### 2. **create_enhanced_comparison_charts.py**
- **Purpose**: Enhanced comparison chart generator
- **Features**:
  - Generate QML/ML style comparison charts
  - Include accuracy comparison, rebalance count comparison, equity curves
- **Output**: `simplified_ultimate_comparison/` directory

#### 3. **create_improved_comparison_charts.py**
- **Purpose**: Improved chart generator
- **Features**:
  - Add model name labels to efficiency analysis charts
  - Remove normalization from performance heatmaps
  - Sort rebalance charts from largest to smallest
- **Output**: Updates `simplified_ultimate_comparison/` directory

#### 4. **create_enhanced_efficiency_analysis.py**
- **Purpose**: Final efficiency analysis generator
- **Features**:
  - Add Sharpe ratio reference lines (0.5, 1.0, 1.5, 2.0) to risk-return charts
  - Generate most complete efficiency analysis charts
- **Output**: Updates `simplified_ultimate_comparison/` directory

### 📈 Extended Scripts

#### 5. **create_steer_ml_comparison.py**
- **Purpose**: Steer strategy vs ML/QML model comparison
- **Features**: Generate comparison charts between steer strategies and machine learning models

#### 6. **create_detailed_steer_ml_comparison.py**
- **Purpose**: Detailed Steer vs ML comparison
- **Features**: Generate more detailed comparison analysis including APR, performance curves

#### 7. **create_comprehensive_steer_comparison.py**
- **Purpose**: Comprehensive steer strategy comparison
- **Features**: Analyze all 16 steer strategies' comprehensive performance

#### 8. **create_simplified_ultimate_comparison.py**
- **Purpose**: Simplified ultimate comparison
- **Features**: Integrate all comparison results, generate final rankings and reports

### 📋 Report Generation Scripts

#### 9. **create_final_enhanced_report.py**
- **Purpose**: Final enhanced report generator
- **Features**: Generate complete analysis reports including all charts

#### 10. **create_final_summary_report.py**
- **Purpose**: Final summary report generator
- **Features**: Generate simplified summary reports

#### 11. **create_summary_visualization.py**
- **Purpose**: Summary visualization generator
- **Features**: Generate simplified visualization charts

### 🧪 Test Scripts

#### 12. **test_steer_fix.py**
- **Purpose**: Steer strategy fix testing
- **Features**: Verify basic functionality after steer strategy fixes

#### 13. **test_steer_detailed.py**
- **Purpose**: Detailed steer strategy testing
- **Features**: Comprehensive testing of steer strategies with various configurations

## 🚀 Usage Guide

### Basic Usage Flow

1. **Fix and Basic Comparison**:
   ```bash
   python run_steer_comparison.py
   ```

2. **Generate Enhanced Charts**:
   ```bash
   python create_enhanced_comparison_charts.py
   ```

3. **Generate Improved Charts**:
   ```bash
   python create_improved_comparison_charts.py
   ```

4. **Generate Final Efficiency Analysis**:
   ```bash
   python create_enhanced_efficiency_analysis.py
   ```

### Complete Analysis Flow

```bash
# 1. Basic Analysis
python run_steer_comparison.py

# 2. Enhanced Comparison
python create_enhanced_comparison_charts.py

# 3. Improved Charts
python create_improved_comparison_charts.py

# 4. Final Efficiency Analysis
python create_enhanced_efficiency_analysis.py

# 5. Generate Final Report
python create_final_enhanced_report.py
```

### One-Command Analysis

```bash
# Run all analysis in sequence
python run_all_analysis.py
```

## 📁 Output Directories

- `steer_comparison_results/` - Basic steer strategy comparison results
- `simplified_ultimate_comparison/` - Ultimate comparison results
- `steer_ml_comparison_results/` - Steer vs ML comparison results
- `detailed_steer_ml_comparison/` - Detailed Steer vs ML comparison results
- `comprehensive_steer_comparison/` - Comprehensive steer strategy comparison results

## 🎯 Main Chart Types

1. **Accuracy Comparison Charts** - Bar charts + Box plots
2. **Rebalance Count Comparison Charts** - Bar charts + Box plots
3. **Equity Curves with Uncertainty Bands** - Time series charts
4. **Performance Heatmaps** - Normalized/Non-normalized heatmaps
5. **Efficiency Analysis Charts** - Four-panel scatter plot analysis
6. **Strategy Ranking Tables** - Comprehensive performance rankings

## 📊 Chart Features

- **English Titles and Labels** - All charts use English
- **Color Coding** - Color-coded by strategy type
- **High Resolution** - 300 DPI output quality
- **Professional Format** - Academic and business report standards
- **Reference Lines** - Include Sharpe ratio and other key indicator reference lines

## 🔧 Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- pathlib
- logging
- warnings

## 📝 Notes

1. Ensure all dependencies are installed before running scripts
2. Some scripts require running prerequisite scripts first
3. Output directories are created automatically
4. All charts are saved in corresponding result directories

## 🎯 Strategy Types Analyzed

### Fixed Strategies
- **Fixed (Conservative)**: 5% width, uniform distribution, capital preservation
- **Fixed (Moderate)**: 10% width, uniform distribution, balanced approach
- **Original (Before Fix)**: Original implementation with cash depletion issues

### ML Strategies
- **ML Bollinger**: Random Forest + Bollinger Bands (70% ML + 30% traditional)
- **ML Keltner**: Random Forest + Keltner Channels
- **ML Donchian**: Random Forest + Donchian Channels

### Quantum Strategies
- **Quantum Bollinger**: QNN + Bollinger Bands with quantum confidence adjustment
- **Quantum Keltner**: QNN + Keltner Channels
- **Quantum Hybrid**: Combined quantum models for intent and price prediction

### Steer Intent Strategies
- **Classic Strategy**: Base strategy with fixed parameters
- **Channel Multiplier**: Single symmetric percentage width around current price
- **Bollinger Strategy**: Traditional Bollinger Bands implementation
- **Keltner Strategy**: Traditional Keltner Channels implementation
- **Donchian Strategy**: Traditional Donchian Channels implementation
- **Stable Strategy**: Multi-position strategy around calculated anchor
- **Fluid Strategy**: Maintains value ratio towards ideal_ratio
- **Imperfect Classic**: Classic strategy with imperfect execution
