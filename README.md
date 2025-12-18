# 🌍 Tourist Arrival Prediction

Machine learning-based time series model to Prediction tourist arrivals using historical patterns.

## 🎯 Problem Statement

Predict tourist arrivals based on historical trends, seasonality, location, weather, and events.

**Problem Type**: Time Series Regression

## 💼 Business Use Case

- Tourism demand planning for government policy
- Hotel & airline capacity forecasting
- Infrastructure and resource allocation
- Revenue optimization strategies

## 📊 Dataset

Time-series dataset with multiple factors affecting tourist arrivals.

**Features**:
- Date, tourist area, facility type
- Tourism index, weather index
- Location attributes
- Event indicators

**Target**: Tourist arrivals (number)

**Size**: 132,191 training records

## 🔬 Approach

1. Data cleaning & preprocessing
2. Time-based feature engineering (year, month, quarter)
3. Lag features (previous periods)
4. Rolling statistics (moving averages)
5. Train-test split using time order
6. Model training & evaluation

## 🤖 Models

- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost

## 📈 Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² score

## 🎯 Results

The model successfully captured trend and seasonality with low prediction error.

## 📁 Project Structure
```
tourist_arrival_prediction/
├── data/
│   ├── raw/          # Original dataset
│   └── processed/    # Cleaned data
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── train.py
├── models/           # Trained models
├── app/              # Streamlit app
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Streamlit

## 🚀 Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python src/data_processing.py
python src/feature_engineering.py
python src/train.py

# Run web app
streamlit run app/app.py
```

## 📄 License

MIT License

