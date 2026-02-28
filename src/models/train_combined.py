import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ingestion.fetch_data import DataFetcher
from src.features.technical_indicators import TechnicalIndicators
from src.models.train import ModelTrainer
from src.utils.logger import setup_logger
from src.utils.helpers import ensure_dir

logger = setup_logger(__name__)

# Hyperparameter configurations to test
TRAINING_PLANS = {
    'baseline': {
        'model_type': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
    },
    'deep_forest': {
        'model_type': 'random_forest',
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
    },
    'wide_forest': {
        'model_type': 'random_forest',
        'n_estimators': 500,
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
    },
    'balanced_forest': {
        'model_type': 'random_forest',
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
    },
    'gradient_boost': {
        'model_type': 'gradient_boosting',
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'min_samples_split': 5,
    },
    'gradient_boost_slow': {
        'model_type': 'gradient_boosting',
        'n_estimators': 500,
        'max_depth': 3,
        'learning_rate': 0.05,
        'min_samples_split': 10,
    },
}


def load_and_prepare_ticker(fetcher: DataFetcher, ticker: str) -> pd.DataFrame:
    """Load data for a ticker and add technical indicators."""
    try:
        data = fetcher.load_raw_data(f"{ticker}_raw.csv")
        
        # Add technical indicators
        indicators = TechnicalIndicators(data)
        indicators.add_sma(window=20).add_sma(window=50)
        indicators.add_rsi().add_returns()
        df = indicators.get_dataframe()
        
        # Add ticker column for identification
        df['Ticker'] = ticker
        
        return df
    except Exception as e:
        logger.warning(f"Could not load {ticker}: {e}")
        return pd.DataFrame()


def train_combined_model():
    """Train a model using data from multiple tickers."""
    logger.info("="*50)
    logger.info("Training Combined Model with Multiple Tickers")
    logger.info("="*50)
    
    fetcher = DataFetcher()
    
    tickers = [
        "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", 
        "NVDA", "META", "TSLA", "BRK-B", "JPM",
        "V", "UNH", "XOM", "JNJ", "WMT",
        "MA", "PG", "HD", "CVX", "MRK"
    ]
    
    # Load and prepare all tickers
    logger.info("\n[1/4] Loading and preparing data for all tickers")
    all_data = []
    for ticker in tickers:
        df = load_and_prepare_ticker(fetcher, ticker)
        if not df.empty:
            all_data.append(df)
            logger.info(f"  Loaded {ticker}: {len(df)} records")
    
    if not all_data:
        logger.error("No data available!")
        return
    
    # Combine all ticker data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"\nCombined dataset: {combined_df.shape}")
    
    # Create target for each ticker
    logger.info("\n[2/4] Creating targets")
    trainer = ModelTrainer(config={'n_estimators': 200, 'max_depth': 15})
    combined_df = trainer.create_target(combined_df)
    
    # Prepare features (excluding non-numeric and identifier columns)
    logger.info("\n[3/4] Preparing features and training")
    X, y = trainer.prepare_data(combined_df, drop_cols=['Date', 'target', 'future_return', 'Ticker'])
    
    # Train/test split (temporal within each ticker would be better, but this is simpler)
    X_train, X_test, y_train, y_test = trainer.train_test_split_temporal(X, y, test_size=0.2)
    
    # Train model
    model = trainer.train(X_train, y_train, model_type='random_forest')
    
    # Evaluate
    logger.info("\n[4/4] Evaluating model")
    
    X_test_scaled = trainer.scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    logger.info(f"\nTest Results on {len(y_test)} samples:")
    logger.info(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    logger.info(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    logger.info(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Save model
    ensure_dir('models')
    trainer.save('models/combined_model.pkl', 'models/combined_scaler.pkl')
    
    logger.info("\n" + "="*50)
    logger.info("Combined Model Training Completed")
    logger.info("="*50)
    
    return trainer, model


def train_with_hyperparameters(X_train, y_train, X_test, y_test, plan_name: str, config: dict):
    """Train a model with specific hyperparameters and return metrics."""
    logger.info(f"\n--- Training Plan: {plan_name} ---")
    logger.info(f"Config: {config}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_type = config.get('model_type', 'random_forest')
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 10),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            class_weight=config.get('class_weight', None),
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 5),
            learning_rate=config.get('learning_rate', 0.1),
            min_samples_split=config.get('min_samples_split', 2),
            random_state=42
        )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'plan': plan_name,
        'model_type': model_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    
    return metrics, model, scaler


def run_hyperparameter_experiments():
    """Run training experiments with different hyperparameter configurations."""
    logger.info("="*60)
    logger.info("HYPERPARAMETER TUNING EXPERIMENTS")
    logger.info("="*60)
    
    fetcher = DataFetcher()
    
    tickers = [
        "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", 
        "NVDA", "META", "TSLA", "BRK-B", "JPM",
        "V", "UNH", "XOM", "JNJ", "WMT",
        "MA", "PG", "HD", "CVX", "MRK"
    ]
    
    # Load and prepare all tickers
    logger.info("\n[1/3] Loading data...")
    all_data = []
    for ticker in tickers:
        df = load_and_prepare_ticker(fetcher, ticker)
        if not df.empty:
            all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset: {combined_df.shape}")
    
    # Prepare data
    logger.info("\n[2/3] Preparing features...")
    trainer = ModelTrainer()
    combined_df = trainer.create_target(combined_df)
    X, y = trainer.prepare_data(combined_df, drop_cols=['Date', 'target', 'future_return', 'Ticker'])
    X_train, X_test, y_train, y_test = trainer.train_test_split_temporal(X, y, test_size=0.2)
    
    # Run experiments
    logger.info("\n[3/3] Running hyperparameter experiments...")
    results = []
    best_f1 = 0
    best_model = None
    best_scaler = None
    best_plan = None
    
    for plan_name, config in TRAINING_PLANS.items():
        metrics, model, scaler = train_with_hyperparameters(
            X_train, y_train, X_test, y_test, plan_name, config
        )
        results.append(metrics)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model
            best_scaler = scaler
            best_plan = plan_name
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT RESULTS SUMMARY")
    logger.info("="*60)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1', ascending=False)
    
    logger.info("\nAll Results (sorted by F1):")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['plan']:20s} | Acc: {row['accuracy']:.4f} | Prec: {row['precision']:.4f} | Rec: {row['recall']:.4f} | F1: {row['f1']:.4f}")
    
    logger.info(f"\nBest Model: {best_plan} (F1: {best_f1:.4f})")
    
    # Save best model
    ensure_dir('models')
    from src.utils.helpers import save_model
    save_model(best_model, 'models/best_model.pkl')
    save_model(best_scaler, 'models/best_scaler.pkl')
    logger.info(f"Best model saved to models/best_model.pkl")
    
    # Save results to CSV
    results_df.to_csv('models/hyperparameter_results.csv', index=False)
    logger.info("Results saved to models/hyperparameter_results.csv")
    
    return results_df, best_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train combined model')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning experiments')
    args = parser.parse_args()
    
    if args.tune:
        run_hyperparameter_experiments()
    else:
        train_combined_model()
