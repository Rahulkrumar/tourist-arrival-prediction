"""
Data Processing Module
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process tourist arrival data"""
    
    def __init__(self, train_path='data/raw/train_df.csv'):
        self.train_path = train_path
        self.df = None
        
    def load_data(self):
        """Load dataset"""
        logger.info("Loading data...")
        self.df = pd.read_csv(self.train_path)
        logger.info(f"Loaded {len(self.df)} records")
        return self
    
    def convert_dates(self):
        """Convert date to datetime"""
        logger.info("Converting dates...")
        self.df['date'] = pd.to_datetime(self.df['date'])
        return self
    
    def sort_data(self):
        """Sort by date"""
        logger.info("Sorting by date...")
        self.df = self.df.sort_values('date').reset_index(drop=True)
        return self
    
    def handle_missing(self):
        """Handle missing values"""
        logger.info("Handling missing values...")
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(0, inplace=True)
        return self
    
    def remove_outliers(self):
        """Remove outliers"""
        logger.info("Removing outliers...")
        if 'tourist_arrivals' in self.df.columns:
            self.df = self.df[self.df['tourist_arrivals'] >= 0]
            q99 = self.df['tourist_arrivals'].quantile(0.995)
            self.df = self.df[self.df['tourist_arrivals'] <= q99]
        return self
    
    def process_all(self, save_path='data/processed'):
        """Run complete pipeline"""
        logger.info("="*60)
        logger.info("DATA PROCESSING")
        logger.info("="*60)
        
        self.load_data()
        self.convert_dates()
        self.sort_data()
        self.handle_missing()
        self.remove_outliers()
        
        # Save
        Path(save_path).mkdir(parents=True, exist_ok=True)
        output = Path(save_path) / 'processed_train.csv'
        self.df.to_csv(output, index=False)
        logger.info(f"Saved to: {output}")
        
        logger.info("="*60)
        return self.df


if __name__ == "__main__":
    processor = DataProcessor('data/raw/train_df.csv')
    df = processor.process_all()
    print(f"\nProcessed: {df.shape}")
