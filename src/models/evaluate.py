import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import load_model

logger = setup_logger(__name__)


class ModelEvaluator:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model = load_model(model_path) if model_path else None
        self.scaler = load_model(scaler_path) if scaler_path else None
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        X_test_scaled = self.scaler.transform(X_test)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info("Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        logger.info("\nConfusion Matrix:")
        logger.info("\n" + str(confusion_matrix(y_test, y_pred)))
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities


if __name__ == "__main__":
    from src.ingestion.fetch_data import DataFetcher
    from src.features.technical_indicators import TechnicalIndicators
    from src.models.train import ModelTrainer
    
    fetcher = DataFetcher()
    data = fetcher.load_raw_data("SPY_raw.csv")
    
    indicators = TechnicalIndicators(data)
    indicators.add_sma(window=20).add_sma(window=50).add_rsi().add_returns()
    df_with_features = indicators.get_dataframe()
    
    trainer = ModelTrainer()
    df_with_target = trainer.create_target(df_with_features)
    X, y = trainer.prepare_data(df_with_target)
    X_train, X_test, y_train, y_test = trainer.train_test_split_temporal(X, y)
    
    evaluator = ModelEvaluator('models/model.pkl', 'models/scaler.pkl')
    metrics = evaluator.evaluate(X_test, y_test)
