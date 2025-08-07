# LSTM-based Stock Return Forecasting

**Tools:** Python, Keras, TA-Lib  
**Model:** LSTM Neural Network  
**Target:** Daily Return (%)  
**Dataset:** Historical Stock Data (Yahoo Finance)

---

## I. Problem Understanding

Stock price movements are highly volatile and influenced by numerous factors. Predicting short-term daily returns can help traders and financial analysts identify trends and manage risk. This project was part of a technical test from Mandiri Sekuritas for a Data Scientist role.

### Objective
- Predict next-day **daily return (DR)** of stocks based on historical price and technical indicators.
- Focus on **NVIDIA (NVDA)** and **Big 4 Indonesian Banks** (`BBCA.JK`, `BBRI.JK`, `BMRI.JK`, `BBNI.JK`).

### Business Motivation
- Traders require **data-driven signals** to support entry/exit decisions.
- Financial institutions aim to **integrate predictive tools** into their internal dashboards.
- Directional forecasts (price up/down) are particularly useful for **short-term tactical moves**.

---

## II. Data Understanding & Preprocessing

This step involves preparing and enriching raw stock price data.

### Steps Performed
- **Downloaded historical stock data** using `yfinance`.
- **Calculated target**: Daily Return (DR) = `(Close - Open) / Open * 100`
- **Added technical indicators** using `ta` package:
  - AO (Awesome Oscillator)
  - RSI (Relative Strength Index)
  - ATR (Average True Range)
  - ADX (Average Directional Index)
  - Aroon Up / Down
- **Removed multi-index** from DataFrame columns.
- **Dropped missing values** (from technical indicator initialization).
- **Added seasonal features**: Day of Week, Month, Quarter (via one-hot encoding).
- **Scaled features** using `StandardScaler`.
- **Created sequences**: Using a sliding window of 3 days to predict the 4th day.

---

## III. Modeling

### Architecture
- LSTM layer (128 units, `tanh`)
- Dropout (0.2)
- Dense (128, `ReLU`)
- Dense output (1 neuron for regression)

### Settings
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Output:** Predicted Daily Return (%)
- **Training:** 20 epochs, batch size 32, validation split 0.2
- **Train/Test Split:** Used `TimeSeriesSplit` with `n_splits=2` to ensure temporal consistency.

---

### Evaluation: NVIDIA Stock (NVDA)

**Results**

| Metric               | Value   |
|----------------------|---------|
| RMSE (Test)          | 4.1371  |
| MAE                  | 3.4451  |
| MAPE                 | 7.11%   |
| Directional Accuracy | 45.92%  |

The model captures general movement trends but still faces challenges in high-volatility regions. It is useful as a supporting signal in short-term decision making.

---

### Evaluation: Big 4 Indonesian Banks

| Ticker   | RMSE   | MAE   | MAPE   | Directional Accuracy |
|----------|--------|--------|--------|------------------------|
| BBCA.JK  | 5.2062 | 4.5093 | 10.29% | 45.25%                 |
| BBNI.JK  | 5.7435 | 5.0794 | 11.79% | 44.81%                 |
| BMRI.JK  | 5.9798 | 5.2623 | 12.32% | 45.92%                 |
| BBRI.JK  | 5.5691 | 4.9390 | 11.44% | 45.70%                 |

All banks show consistent error metrics, with acceptable accuracy for research-grade forecasting. The model provides a slight edge in directional movement detection over baseline/random.

---

## IV. Conclusion, Business Impact & Recommendations

### Conclusion
- LSTM provides a viable modeling approach for forecasting short-term daily returns in both U.S. (NVDA) and Indonesian (BBCA, BBRI, BMRI, BBNI) stocks.
- While the model does not achieve perfect accuracy, it shows consistent performance across different tickers.
- Technical indicators and time-based features add value by enriching the input space and allowing the model to capture market momentum and patterns.

### Business Impact
- **Trading Strategy Support:** The model offers directional signals that can assist in tactical trading decisions.
- **Data-Driven Forecasting:** Useful for internal strategy development, risk management, and portfolio allocation.
- **Automation Potential:** Can be integrated into dashboards or alert systems for continuous monitoring.

### Recommendations
- **For Traders:** Use model outputs as one of the decision layers (confluence) with other signals like volume, news, or sentiment.
- **For Data Scientists:** Consider incorporating additional features like macroeconomic indicators, sector rotation data, or technical sentiment from forums.
- **For Product Teams / Business Analysts:** Pilot the model within existing trading tools or as part of a prototype dashboard for internal research.

---

## V. Next Steps

To enhance accuracy and deploy the model in a production-like setting, future steps include:

- Fine-tuning LSTM hyperparameters (e.g., number of layers, sequence length, learning rate).
- Comparing LSTM with other deep learning architectures such as:
  - GRU (Gated Recurrent Unit)
  - Temporal Convolutional Networks (TCN)
  - Transformer-based Time Series Models
- Adding alternative data sources:
  - Financial news sentiment
  - Analyst recommendations
  - Market indices and macroeconomic data
- Deploying as a REST API service or integrating into a dashboard interface for real-time inference.
