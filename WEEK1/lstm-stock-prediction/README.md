# 📈 Day 5 – LSTM Stock Price Prediction (Real Data)

This project applies **Long Short-Term Memory (LSTM)** networks to real-world **stock price forecasting** using historical data from Yahoo Finance.

## 🚀 Project Overview
- **Goal:** Predict future stock prices from historical data.
- **Dataset:** Yahoo Finance stock prices (AAPL by default).
- **Model:** Multi-layer LSTM with dropout.
- **Evaluation:** Compare predictions vs. actual prices on test data.

## 📊 Workflow
1. Download data from Yahoo Finance using `yfinance`.
2. Preprocess (scaling + sequence creation).
3. Train/test split.
4. Build LSTM model in Keras/TensorFlow.
5. Train & evaluate model.
6. Visualize predictions.

## ⚙️ Requirements
```bash
pip install -r requirements.txt
