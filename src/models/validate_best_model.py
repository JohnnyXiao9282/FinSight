import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.fetch_data import DataFetcher
from src.features.technical_indicators import TechnicalIndicators
from src.models.train import ModelTrainer
from src.utils.helpers import load_model
from src.utils.logger import setup_logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

logger = setup_logger(__name__)

if __name__ == "__main__":
    # Load best model and scaler
    model = load_model('models/best_model.pkl')
    scaler = load_model('models/best_scaler.pkl')
    
    # Use SPY as unseen validation ticker
    fetcher = DataFetcher()
    data = fetcher.load_raw_data('SPY_raw.csv')
    indicators = TechnicalIndicators(data)
    indicators.add_sma(window=20).add_sma(window=50).add_rsi().add_returns()
    df = indicators.get_dataframe()
    trainer = ModelTrainer()
    df = trainer.create_target(df)
    X, y = trainer.prepare_data(df, drop_cols=['Date', 'target', 'future_return'])
    
    # Use last 20% as validation
    split_idx = int(len(X) * 0.8)
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
    X_val_scaled = scaler.transform(X_val)
    y_pred = model.predict(X_val_scaled)
    
    logger.info(f"Validation Results on SPY ({len(y_val)} samples):")
    logger.info(f"  Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    logger.info(f"  Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
    logger.info(f"  Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
    logger.info(f"  F1 Score:  {f1_score(y_val, y_pred, zero_division=0):.4f}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_val, y_pred))
