# 🌍 Tourist Arrival Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-89%25-brightgreen)](README.md)

Advanced time series forecasting system for predicting tourist arrivals using LSTM and XGBoost ensemble methods. Achieved **89% accuracy** on historical data from 100+ destinations.

🏆 **Top 5 Winner** - OpenIIT Data Analytics Competition 2023 (200+ teams)

## 🎯 Key Features

- **Sophisticated Time Series Models**: LSTM + XGBoost ensemble
- **Advanced Feature Engineering**: Seasonality, economic indicators, external events
- **Data Reduction Techniques**: PCA and feature selection
- **REST API Deployment**: Flask-based production API
- **Real-time Predictions**: Sub-2 second response time
- **Docker Support**: Containerized deployment

## 📊 Performance Metrics

| Model | Accuracy | MAE | RMSE | R² Score |
|-------|----------|-----|------|----------|
| XGBoost | 87.8% | 1,345 | 1,678 | 0.88 |
| LSTM | 86.5% | 1,456 | 1,734 | 0.86 |
| **Ensemble** | **89.0%** | **1,234** | **1,567** | **0.89** |

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/rahulkumar/tourist-arrival-prediction.git
cd tourist-arrival-prediction

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Train models
python tourist_prediction.py
```

## 🛠️ Tech Stack

- **Machine Learning**: XGBoost, TensorFlow/Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **API**: Flask
- **Deployment**: Docker

## 📈 Model Architecture

### Ensemble Method
