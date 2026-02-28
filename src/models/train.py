import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from src.utils.helpers import save_model

logger = setup_logger(__name__)


class ModelTrainer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_target(self, df: pd.DataFrame, column: str = 'Close', horizon: int = 1) -> pd.DataFrame:
        df = df.copy()
        df['future_return'] = df[column].pct_change(periods=horizon).shift(-horizon)
        df['target'] = (df['future_return'] > 0).astype(int)
        logger.info(f"Created target with {df['target'].sum()} positive samples")
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target', drop_cols: List[str] = None) -> Tuple:
        df = df.dropna()
        
        if drop_cols is None:
            drop_cols = ['Date', 'target', 'future_return']
        
        # Handle both single-level and multi-level column names
        def should_drop(col):
            if isinstance(col, tuple):
                return any(d in col for d in drop_cols)
            return col in drop_cols
        
        feature_cols = [col for col in df.columns if not should_drop(col)]
        
        X = df[feature_cols]
        
        # Keep only numeric columns (exclude datetime)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        y = df[target_col]
        
        self.feature_names = numeric_cols
        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_test_split_temporal(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'random_forest'):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 10),
                random_state=self.config.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=self.config.get('max_iter', 1000),
                random_state=self.config.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Training {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        logger.info("Training completed")
        
        return self.model
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X_scaled = self.scaler.fit_transform(X)
        
        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train_fold, y_train_fold)
            score = self.model.score(X_val_fold, y_val_fold)
            scores.append(score)
            logger.info(f"Fold {fold + 1}/{n_splits}: Accuracy = {score:.4f}")
        
        mean_score = np.mean(scores)
        logger.info(f"Mean CV Score: {mean_score:.4f} (+/- {np.std(scores):.4f})")
        
        return scores
    
    def save(self, model_path: str, scaler_path: str):
        save_model(self.model, model_path)
        save_model(self.scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    from src.ingestion.fetch_data import DataFetcher
    from src.features.technical_indicators import TechnicalIndicators
    
    fetcher = DataFetcher()
    data = fetcher.load_raw_data("SPY_raw.csv")
    
    indicators = TechnicalIndicators(data)
    indicators.add_sma(window=20).add_rsi().add_returns()
    df_with_features = indicators.get_dataframe()
    
    trainer = ModelTrainer()
    df_with_target = trainer.create_target(df_with_features)
    X, y = trainer.prepare_data(df_with_target)
    X_train, X_test, y_train, y_test = trainer.train_test_split_temporal(X, y)
    
    trainer.train(X_train, y_train, model_type='random_forest')
    trainer.save('models/model.pkl', 'models/scaler.pkl')
