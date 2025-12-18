"""
Feature Engineering Module
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for prediction"""
    
    def __init__(self, df):
        self.df = df.copy()
        
    def time_features(self):
        """Time-based features"""
        logger.info("Creating time features...")
        df = self.df
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        self.df = df
        return self
    
    def lag_features(self):
        """Lag features"""
        logger.info("Creating lag features...")
        df = self.df.sort_values(['tourist_area', 'spot_facility', 'date'])
        
        for lag in [1, 3, 6, 12]:
            df[f'lag_{lag}'] = df.groupby(['tourist_area', 'spot_facility'])['tourist_arrivals'].shift(lag)
        
        self.df = df
        return self
    
    def rolling_features(self):
        """Rolling statistics"""
        logger.info("Creating rolling features...")
        df = self.df
        
        for window in [3, 6, 12]:
            df[f'rolling_mean_{window}'] = (
                df.groupby(['tourist_area', 'spot_facility'])['tourist_arrivals']
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
        
        self.df = df
        return self
    
    def aggregated_features(self):
        """Aggregations"""
        logger.info("Creating aggregated features...")
        df = self.df
        df['area_mean'] = df.groupby('tourist_area')['tourist_arrivals'].transform('mean')
        df['month_mean'] = df.groupby('month')['tourist_arrivals'].transform('mean')
        self.df = df
        return self
    
    def encode_categorical(self):
        """Encode categorical"""
        logger.info("Encoding categorical...")
        df = self.df
        for col in ['area', 'city', 'type', 'info', 'event']:
            if col in df.columns:
                df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
        self.df = df
        return self
    
    def create_all(self):
        """Create all features"""
        logger.info("="*60)
        logger.info("FEATURE ENGINEERING")
        logger.info("="*60)
        
        self.time_features()
        if 'tourist_arrivals' in self.df.columns:
            self.lag_features()
            self.rolling_features()
            self.aggregated_features()
        self.encode_categorical()
        
        logger.info(f"Total features: {len(self.df.columns)}")
        logger.info("="*60)
        return self.df
    
    def save(self, path='data/processed/features.csv'):
        """Save features"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)
        logger.info(f"Saved to: {path}")


if __name__ == "__main__":
    df = pd.read_csv('data/processed/processed_train.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    engineer = FeatureEngineer(df)
    df_features = engineer.create_all()
    engineer.save('data/processed/features.csv')
    
    print(f"\nFeatures: {df_features.shape}")
