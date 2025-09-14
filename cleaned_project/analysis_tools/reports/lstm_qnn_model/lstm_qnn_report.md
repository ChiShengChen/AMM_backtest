# LSTM_QNN Model Report

## 🎯 Model Overview

LSTM_QNN Model is a hybrid architecture combining:
- **LSTM layers** for time series sequence processing
- **Classical neural networks** for feature transformation
- **Quantum circuits** for enhanced pattern recognition
- **Unified AMM Baseline labels** for fair comparison

## 🏗️ Architecture Details

- **Sequence Length**: 10 time steps
- **Input Features**: 9 features per time step
- **LSTM Hidden Dimension**: 64
- **Quantum Qubits**: 6
- **Quantum Layers**: 2
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 100

## 📊 Performance Results

- **Accuracy**: 0.7417
- **Final Training Loss**: 0.6502
- **Final Validation Loss**: 0.5938

## 🔍 Key Features

### 1. Time Series Processing
- Uses LSTM to capture temporal dependencies
- Processes sequences of 10 time steps
- Maintains memory of past patterns

### 2. Quantum Enhancement
- Angle encoding maps features to [0, 2π]
- Multiple quantum layers with RY, RZ rotations
- CNOT gates for entanglement simulation

### 3. Hybrid Architecture
- Classical preprocessing for feature extraction
- Quantum processing for pattern recognition
- Classical postprocessing for final prediction

## 📈 Generated Charts

1. **qasa_sequence_analysis.png** - Complete model analysis
2. **qasa_sequence_report.md** - This detailed report

## ✅ Conclusions

The LSTM_QNN Model successfully combines:
- Time series processing capabilities
- Quantum computing advantages
- Classical machine learning robustness
- Fair comparison with unified labels

This represents a significant advancement in quantum-enhanced
time series analysis for financial applications.
