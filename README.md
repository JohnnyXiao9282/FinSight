# FinSight

Financial Machine Learning Pipeline for Market Regime Classification

## Overview

FinSight is a production-ready ML pipeline for predicting market regimes using technical indicators and time-series features. Built with proper separation of concerns, time-aware validation, and API-ready inference.

## Features

- 📊 **Data Ingestion**: Automated financial data download via yfinance (20+ tickers supported)
- ✅ **Data Validation**: Quality checks, missing value detection, outlier analysis
- 🔧 **Feature Engineering**: Technical indicators (RSI, SMA, EMA, MACD, Bollinger Bands)
- 🤖 **Model Training**: Multiple models with hyperparameter tuning (Random Forest, Gradient Boosting)
- 📈 **Backtesting**: Walk-forward validation to prevent data leakage
- 🚀 **REST API**: FastAPI inference endpoint for predictions
- 📊 **PowerBI Integration**: Export predictions and profitability reports for dashboards
- 📝 **Logging**: All operations logged to `logs/finsight.log`

## Project Structure

```
finsight/
├── data/                    # Raw and processed data (20 tickers)
├── configs/                 # Configuration files
│   └── config.yaml
├── logs/                    # Log files
│   └── finsight.log
├── models/                  # Trained models and results
│   ├── best_model.pkl       # Best performing model
│   ├── best_scaler.pkl      # Scaler for best model
│   ├── combined_model.pkl   # Multi-ticker model
│   └── hyperparameter_results.csv
├── src/
│   ├── ingestion/           # Data fetching
│   ├── validation/          # Data quality checks
│   ├── features/            # Feature engineering
│   ├── models/              # Training and evaluation
│   │   ├── train.py
│   │   ├── train_combined.py    # Multi-ticker training with hyperparameter tuning
│   │   ├── evaluate.py
│   │   └── validate_best_model.py
│   ├── backtesting/         # Walk-forward validation
│   ├── pipeline/            # End-to-end orchestration
│   ├── api/                 # FastAPI inference server
│   └── utils/               # Logging and helpers
├── generate_powerbi_report.py   # PowerBI dashboard data generator
├── export_for_powerbi.py        # Export results for PowerBI
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch Data (20 Tickers)

```bash
python src/ingestion/fetch_data.py
```

This fetches data for: SPY, AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V, UNH, XOM, JNJ, WMT, MA, PG, HD, CVX, MRK

### 3. Run the Pipeline

```bash
python src/pipeline/pipeline.py
```

### 4. Train Combined Model with Hyperparameter Tuning

```bash
# Default training
python src/models/train_combined.py

# With hyperparameter tuning experiments
python src/models/train_combined.py --tune
```

**Hyperparameter Plans Tested:**
- baseline: n_estimators=100, max_depth=10
- deep_forest: n_estimators=200, max_depth=20
- wide_forest: n_estimators=500, max_depth=8 (Best F1: 68.5%)
- balanced_forest: With class_weight='balanced'
- gradient_boost: GradientBoosting with learning_rate=0.1
- gradient_boost_slow: GradientBoosting with learning_rate=0.05

### 5. Validate Best Model

```bash
python src/models/validate_best_model.py
```

### 6. Run Backtesting

```bash
python src/backtesting/backtest.py
```

### 7. Generate PowerBI Reports

```bash
python generate_powerbi_report.py --ticker SPY --capital 10000
```

**Output Files:**
- `powerbi_predictions.csv` - Daily predictions, signals, returns
- `powerbi_summary.csv` - Overall profitability summary
- `powerbi_monthly.csv` - Monthly performance breakdown

### 8. Start the API

```bash
python src/api/main.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Usage Examples

### Fetch Data

```python
from src.ingestion.fetch_data import DataFetcher

fetcher = DataFetcher()
data = fetcher.fetch_stock_data("SPY", "2020-01-01", "2024-12-31")
fetcher.save_raw_data(data, "SPY_raw.csv")
```

### Engineer Features

```python
from src.features.technical_indicators import TechnicalIndicators

indicators = TechnicalIndicators(data)
indicators.add_sma(window=20).add_rsi().add_macd()
df_features = indicators.get_dataframe()
```

### Train Model

```python
from src.models.train import ModelTrainer

trainer = ModelTrainer()
df_with_target = trainer.create_target(df_features)
X, y = trainer.prepare_data(df_with_target)
X_train, X_test, y_train, y_test = trainer.train_test_split_temporal(X, y)

model = trainer.train(X_train, y_train, model_type='random_forest')
trainer.save('models/model.pkl', 'models/scaler.pkl')
```

### Backtest

```python
from src.backtesting.backtest import Backtester

backtester = Backtester(initial_window=252, step_size=21)
results = backtester.walk_forward_validation(X, y)
```

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Get regime prediction (uses best_model.pkl)

## Model Performance

### Best Model: wide_forest (Random Forest)
| Metric | Value |
|--------|-------|
| Accuracy | 52.5% |
| Precision | 52.5% |
| Recall | 98.5% |
| F1 Score | 68.5% |

### Backtest Results (SPY, 2015-2026)
| Metric | Value |
|--------|-------|
| Initial Capital | $10,000 |
| Final Strategy Value | $65,949 |
| Strategy Return | 559.49% |
| Market Return | 300.19% |
| Outperformance | +259.30% |
| **Profitable** | YES |
| **Beats Market** | YES |

## PowerBI Dashboard

Generate reports for PowerBI visualization:

```bash
python generate_powerbi_report.py --ticker SPY --capital 10000
```

Import the CSV files into PowerBI Desktop:
- Build line charts for portfolio value over time
- Create bar charts for monthly returns
- Add gauges for accuracy and profitability metrics
- Compare strategy vs market performance

## Configuration

Edit `configs/config.yaml` to customize:

- Data sources and date ranges
- Feature engineering parameters
- Model hyperparameters
- Backtesting settings

## Future Optimizations

The following optimizations are planned:

1. **Threshold Tuning**: Adjust decision threshold to improve precision
2. **Hyperparameter Tuning**: Further grid search for optimal parameters
3. **Feature Engineering**: Add volatility, volume spikes, macro indicators

## Next Steps

- Implement advanced features (order flow, sentiment)
- Add XGBoost and neural network models
- Integrate MLflow for experiment tracking
- Add DVC for data versioning
- Deploy with Docker

## License

MIT
