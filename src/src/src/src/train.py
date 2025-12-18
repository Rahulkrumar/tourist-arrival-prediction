"""
Model Training Module
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train prediction models"""
    
    def __init__(self, data_path='data/processed/features.csv'):
        self.data_path = data_path
        self.df = None
        self.models = {}
        
    def load_data(self):
        """Load features"""
        logger.info("Loading data...")
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        return self
    
    def prepare_data(self):
        """Prepare X and y"""
        logger.info("Preparing data...")
        
        exclude = ['tourist_arrivals', 'id', 'date', 'area', 'city', 'type', 'info', 'event', 'spot_facility']
        features = [c for c in self.df.columns if c not in exclude]
        
        # Remove high-null features
        null_pct = self.df[features].isnull().mean()
        features = [f for f in features if null_pct[f] < 0.3]
        
        X = self.df[features].fillna(0)
        y = self.df['tourist_arrivals']
        dates = self.df['date']
        
        # Remove NaN targets
        valid = ~y.isna()
        X, y, dates = X[valid], y[valid], dates[valid]
        
        logger.info(f"Features: {len(features)}")
        logger.info(f"Records: {len(X)}")
        
        return X, y, dates, features
    
    def time_split(self, X, y, dates, split_date='2019-07-01'):
        """Time-based split"""
        logger.info(f"Splitting at: {split_date}")
        split = pd.to_datetime(split_date)
        
        train_mask = dates < split
        test_mask = dates >= split
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        logger.info(f"Train: {len(X_train)}")
        logger.info(f"Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        logger.info("Training models...")
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['linear'] = lr
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        self.models['xgboost'] = xgb
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            mask = y_test != 0
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            
            results[name] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R²': r2}
            
            logger.info(f"\n{name.upper()}:")
            logger.info(f"  MAE: {mae:.2f}")
            logger.info(f"  RMSE: {rmse:.2f}")
            logger.info(f"  MAPE: {mape:.2f}%")
            logger.info(f"  R²: {r2:.4f}")
        
        return results
    
    def save_best(self, results, features):
        """Save best model"""
        best = min(results, key=lambda x: results[x]['MAE'])
        logger.info(f"\nBest model: {best}")
        
        Path('models').mkdir(exist_ok=True)
        joblib.dump(self.models[best], 'models/best_model.pkl')
        joblib.dump(features, 'models/features.pkl')
        logger.info("Model saved!")
    
    def train_all(self):
        """Complete pipeline"""
        logger.info("="*60)
        logger.info("MODEL TRAINING")
        logger.info("="*60)
        
        self.load_data()
        X, y, dates, features = self.prepare_data()
        X_train, X_test, y_train, y_test = self.time_split(X, y, dates)
        self.train_models(X_train, y_train)
        results = self.evaluate(X_test, y_test)
        self.save_best(results, features)
        
        logger.info("="*60)
        return results


if __name__ == "__main__":
    trainer = ModelTrainer('data/processed/features.csv')
    results = trainer.train_all()
    print("\nTraining complete!")
