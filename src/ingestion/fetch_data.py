import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataFetcher:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            
            if data.empty:
                logger.warning(f"No data retrieved for {ticker}")
                return pd.DataFrame()
            
            # Flatten multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[1] == '' or col[1] == ticker else f"{col[0]}_{col[1]}" for col in data.columns]
            
            data.reset_index(inplace=True)
            logger.info(f"Retrieved {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {str(e)}")
            raise
    
    def save_raw_data(self, data: pd.DataFrame, filename: str):
        filepath = self.data_dir / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
    
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = pd.read_csv(filepath)
        logger.info(f"Loaded {len(data)} records from {filepath}")
        return data
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None
    ) -> dict:
        results = {}
        for ticker in tickers:
            data = self.fetch_stock_data(ticker, start_date, end_date)
            results[ticker] = data
            if not data.empty:
                self.save_raw_data(data, f"{ticker}_raw.csv")
        return results


if __name__ == "__main__":
    fetcher = DataFetcher()
    
    tickers = [
        "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", 
        "NVDA", "META", "TSLA", "BRK-B", "JPM",
        "V", "UNH", "XOM", "JNJ", "WMT",
        "MA", "PG", "HD", "CVX", "MRK"
    ]
    
    results = fetcher.fetch_multiple_tickers(
        tickers=tickers,
        start_date="2015-01-01"
    )
    
    for ticker, df in results.items():
        print(f"{ticker}: {df.shape}")
