"""
Tourist Arrival Prediction System
Complete implementation with data reduction and deployment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA GENERATION & LOADING ====================
def generate_sample_data(n_months=60, n_destinations=5):
    """Generate sample tourist arrival data"""
    np.random.seed(42)
    dates = pd.date_range(start='2019-01-01', periods=n_months, freq='M')
    
    data = []
    for dest_id in range(1, n_destinations + 1):
        base_arrivals = np.random.randint(5000, 20000)
        trend = np.linspace(0, 3000, n_months)
        seasonality = 5000 * np.sin(np.linspace(0, 10*np.pi, n_months))
        noise = np.random.normal(0, 1000, n_months)
        
        arrivals = base_arrivals + trend + seasonality + noise
        arrivals = np.maximum(arrivals, 0)
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'destination_id': dest_id,
                'arrivals': int(arrivals[i]),
                'month': date.month,
                'quarter': date.quarter,
                'year': date.year,
                'gdp_growth': np.random.uniform(2, 5),
                'exchange_rate': np.random.uniform(70, 80),
                'holiday_flag': 1 if date.month in [12, 1, 4, 7] else 0
            })
    
    return pd.DataFrame(data)

# ==================== FEATURE ENGINEERING ====================
def create_features(df):
    """Create time series features"""
    df = df.copy()
    df = df.sort_values(['destination_id', 'date'])
    
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f'arrivals_lag_{lag}'] = df.groupby('destination_id')['arrivals'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        df[f'arrivals_rolling_mean_{window}'] = df.groupby('destination_id')['arrivals'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'arrivals_rolling_std_{window}'] = df.groupby('destination_id')['arrivals'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop rows with NaN from lag features
    df = df.dropna()
    
    return df

# ==================== DATA REDUCTION ====================
def reduce_data(df, method='pca', n_components=10):
    """Reduce dimensionality of features"""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_regression
    
    feature_cols = [col for col in df.columns if col not in ['date', 'destination_id', 'arrivals']]
    X = df[feature_cols]
    y = df['arrivals']
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_.sum():.4f}")
        return X_reduced, reducer
    
    elif method == 'select_k_best':
        selector = SelectKBest(score_func=f_regression, k=n_components)
        X_reduced = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        print(f"Selected features: {selected_features}")
        return X_reduced, selector
    
    return X, None

# ==================== XGBOOST MODEL ====================
class XGBoostPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.model = xgb.XGBRegressor(**params)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path='xgb_model.pkl'):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, path='xgb_model.pkl'):
        """Load model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']

# ==================== LSTM MODEL ====================
class LSTMPredictor:
    def __init__(self, lookback=12):
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback = lookback
        
    def prepare_sequences(self, data):
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train LSTM model"""
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_train_scaled)
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        
        # Build model
        self.model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self.prepare_sequences(X_scaled)
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()
    
    def save(self, path='lstm_model.h5'):
        """Save model"""
        self.model.save(path)
        with open('lstm_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

# ==================== ENSEMBLE MODEL ====================
class EnsemblePredictor:
    def __init__(self):
        self.xgb_model = XGBoostPredictor()
        self.lstm_model = LSTMPredictor()
        self.weights = {'xgb': 0.6, 'lstm': 0.4}
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train both models"""
        print("Training XGBoost...")
        self.xgb_model.train(X_train, y_train, X_val, y_val)
        
        print("Training LSTM...")
        self.lstm_model.train(X_train.values.reshape(-1, 1), y_train)
        
        return self
    
    def predict(self, X):
        """Ensemble predictions"""
        xgb_pred = self.xgb_model.predict(X)
        lstm_pred = self.lstm_model.predict(X.values.reshape(-1, 1))
        
        # Align lengths
        min_len = min(len(xgb_pred), len(lstm_pred))
        xgb_pred = xgb_pred[:min_len]
        lstm_pred = lstm_pred[:min_len]
        
        # Weighted average
        ensemble_pred = (self.weights['xgb'] * xgb_pred + 
                        self.weights['lstm'] * lstm_pred)
        return ensemble_pred

# ==================== EVALUATION ====================
def evaluate_model(y_true, y_pred, model_name='Model'):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Accuracy: {100 - mape:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("=" * 70)
    print("TOURIST ARRIVAL PREDICTION SYSTEM")
    print("=" * 70)
    
    # 1. Generate/Load Data
    print("\n1. Loading data...")
    df = generate_sample_data(n_months=60, n_destinations=5)
    print(f"Data shape: {df.shape}")
    
    # 2. Feature Engineering
    print("\n2. Creating features...")
    df_features = create_features(df)
    print(f"Feature shape: {df_features.shape}")
    
    # 3. Prepare data
    feature_cols = [col for col in df_features.columns 
                   if col not in ['date', 'destination_id', 'arrivals']]
    X = df_features[feature_cols]
    y = df_features['arrivals']
    
    # 4. Data Reduction
    print("\n3. Applying data reduction...")
    X_reduced, reducer = reduce_data(df_features, method='select_k_best', n_components=10)
    X_reduced = pd.DataFrame(X_reduced)
    
    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")
    
    # 6. Train Models
    print("\n4. Training models...")
    
    # XGBoost
    xgb_model = XGBoostPredictor()
    xgb_model.train(X_train, y_train, X_test, y_test)
    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = evaluate_model(y_test, xgb_pred, "XGBoost")
    
    # LSTM
    lstm_model = LSTMPredictor(lookback=12)
    lstm_model.train(X_train.values.reshape(-1, 1), y_train.values, epochs=30)
    lstm_pred = lstm_model.predict(X_test.values.reshape(-1, 1))
    
    # Align predictions
    min_len = min(len(y_test), len(lstm_pred))
    lstm_metrics = evaluate_model(y_test.values[:min_len], lstm_pred[:min_len], "LSTM")
    
    # Ensemble
    print("\n5. Training ensemble model...")
    ensemble = EnsemblePredictor()
    ensemble.train(X_train, y_train, X_test, y_test)
    ensemble_pred = ensemble.predict(X_test)
    
    min_len = min(len(y_test), len(ensemble_pred))
    ensemble_metrics = evaluate_model(y_test.values[:min_len], ensemble_pred[:min_len], "Ensemble")
    
    # 7. Save Models
    print("\n6. Saving models...")
    xgb_model.save('xgb_model.pkl')
    print("Models saved successfully!")
    
    # 8. Visualization
    print("\n7. Creating visualizations...")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_test.values[:50], label='Actual', marker='o')
    plt.plot(xgb_pred[:50], label='XGBoost', marker='s', alpha=0.7)
    plt.plot(ensemble_pred[:50], label='Ensemble', marker='^', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Tourist Arrivals')
    plt.title('Predictions vs Actual (First 50 samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    models = ['XGBoost', 'LSTM', 'Ensemble']
    accuracies = [
        100 - xgb_metrics['mape'],
        100 - lstm_metrics['mape'],
        100 - ensemble_metrics['mape']
    ]
    plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Comparison')
    plt.ylim([80, 100])
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved as 'model_performance.png'")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
