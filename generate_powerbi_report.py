import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent))

from src.ingestion.fetch_data import DataFetcher
from src.features.technical_indicators import TechnicalIndicators
from src.models.train import ModelTrainer
from src.utils.helpers import load_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_powerbi_report(ticker: str = "SPY", initial_capital: float = 10000.0):
    """
    Generate predictions and profitability report for PowerBI dashboard.
    
    Args:
        ticker: Stock ticker to analyze
        initial_capital: Starting capital for backtest simulation
    """
    logger.info(f"Generating PowerBI report for {ticker}")
    
    # Load model and scaler
    model = load_model('models/best_model.pkl')
    scaler = load_model('models/best_scaler.pkl')
    
    # Load and prepare data
    fetcher = DataFetcher()
    data = fetcher.load_raw_data(f'{ticker}_raw.csv')
    
    indicators = TechnicalIndicators(data)
    indicators.add_sma(window=20).add_sma(window=50).add_rsi().add_returns()
    df = indicators.get_dataframe()
    
    # Create target
    trainer = ModelTrainer()
    df = trainer.create_target(df)
    
    # Prepare features
    X, y = trainer.prepare_data(df, drop_cols=['Date', 'target', 'future_return'])
    
    # Get dates for the predictions
    df_clean = df.dropna().reset_index(drop=True)
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Build results DataFrame
    results = pd.DataFrame({
        'Date': df_clean['Date'].values[:len(predictions)],
        'Ticker': ticker,
        'Close': df_clean['Close'].values[:len(predictions)],
        'Actual': y.values,
        'Prediction': predictions,
        'Probability': probabilities,
        'Signal': ['BUY' if p == 1 else 'HOLD' for p in predictions],
    })
    
    # Calculate returns
    results['Daily_Return'] = results['Close'].pct_change()
    results['Strategy_Return'] = results['Daily_Return'] * results['Prediction'].shift(1)
    results['Cumulative_Market_Return'] = (1 + results['Daily_Return']).cumprod()
    results['Cumulative_Strategy_Return'] = (1 + results['Strategy_Return'].fillna(0)).cumprod()
    
    # Calculate portfolio value
    results['Market_Portfolio'] = initial_capital * results['Cumulative_Market_Return']
    results['Strategy_Portfolio'] = initial_capital * results['Cumulative_Strategy_Return']
    results['Strategy_PnL'] = results['Strategy_Portfolio'] - initial_capital
    
    # Calculate metrics
    results['Correct'] = (results['Prediction'] == results['Actual']).astype(int)
    results['Rolling_Accuracy_20'] = results['Correct'].rolling(20).mean()
    
    # Export predictions
    results.to_csv('powerbi_predictions.csv', index=False)
    logger.info(f"Exported predictions to powerbi_predictions.csv")
    
    # Summary metrics
    total_trades = (results['Prediction'] == 1).sum()
    correct_predictions = results['Correct'].sum()
    accuracy = correct_predictions / len(results) if len(results) > 0 else 0
    final_strategy_value = results['Strategy_Portfolio'].iloc[-1]
    final_market_value = results['Market_Portfolio'].iloc[-1]
    total_return = (final_strategy_value - initial_capital) / initial_capital * 100
    market_return = (final_market_value - initial_capital) / initial_capital * 100
    
    summary = pd.DataFrame([{
        'Report_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Ticker': ticker,
        'Total_Days': len(results),
        'Total_Trades': total_trades,
        'Correct_Predictions': correct_predictions,
        'Accuracy': round(accuracy, 4),
        'Initial_Capital': initial_capital,
        'Final_Strategy_Value': round(final_strategy_value, 2),
        'Final_Market_Value': round(final_market_value, 2),
        'Strategy_Return_Pct': round(total_return, 2),
        'Market_Return_Pct': round(market_return, 2),
        'Outperformance_Pct': round(total_return - market_return, 2),
        'Profitable': 'Yes' if total_return > 0 else 'No',
        'Beats_Market': 'Yes' if total_return > market_return else 'No',
    }])
    
    summary.to_csv('powerbi_summary.csv', index=False)
    logger.info(f"Exported summary to powerbi_summary.csv")
    
    # Monthly performance
    results['Month'] = pd.to_datetime(results['Date']).dt.to_period('M').astype(str)
    monthly = results.groupby('Month').agg({
        'Strategy_Return': 'sum',
        'Daily_Return': 'sum',
        'Correct': 'sum',
        'Prediction': 'count'
    }).reset_index()
    monthly.columns = ['Month', 'Strategy_Return', 'Market_Return', 'Correct_Predictions', 'Total_Predictions']
    monthly['Monthly_Accuracy'] = monthly['Correct_Predictions'] / monthly['Total_Predictions']
    monthly['Strategy_Return_Pct'] = monthly['Strategy_Return'] * 100
    monthly['Market_Return_Pct'] = monthly['Market_Return'] * 100
    monthly.to_csv('powerbi_monthly.csv', index=False)
    logger.info(f"Exported monthly performance to powerbi_monthly.csv")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PROFITABILITY SUMMARY")
    logger.info("="*50)
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Period: {results['Date'].iloc[0]} to {results['Date'].iloc[-1]}")
    logger.info(f"Total Days: {len(results)}")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Accuracy: {accuracy:.2%}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Final Strategy Value: ${final_strategy_value:,.2f}")
    logger.info(f"Strategy Return: {total_return:.2f}%")
    logger.info(f"Market Return: {market_return:.2f}%")
    logger.info(f"Outperformance: {total_return - market_return:.2f}%")
    logger.info(f"PROFITABLE: {'YES' if total_return > 0 else 'NO'}")
    logger.info(f"BEATS MARKET: {'YES' if total_return > market_return else 'NO'}")
    
    return results, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate PowerBI report')
    parser.add_argument('--ticker', default='SPY', help='Stock ticker')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    args = parser.parse_args()
    
    generate_powerbi_report(ticker=args.ticker, initial_capital=args.capital)
