import pandas as pd
from pathlib import Path

# Export model results for PowerBI dashboard

def export_hyperparameter_results():
    src = Path('models/hyperparameter_results.csv')
    if src.exists():
        df = pd.read_csv(src)
        df.to_csv('powerbi_hyperparameter_results.csv', index=False)
        print('Exported hyperparameter results to powerbi_hyperparameter_results.csv')
    else:
        print('No hyperparameter results found.')

def export_backtest_results():
    src = Path('models/backtest_results.csv')
    if src.exists():
        df = pd.read_csv(src)
        df.to_csv('powerbi_backtest_results.csv', index=False)
        print('Exported backtest results to powerbi_backtest_results.csv')
    else:
        print('No backtest results found.')

def export_predictions():
    src = Path('models/predictions.csv')
    if src.exists():
        df = pd.read_csv(src)
        df.to_csv('powerbi_predictions.csv', index=False)
        print('Exported predictions to powerbi_predictions.csv')
    else:
        print('No predictions found.')

if __name__ == "__main__":
    export_hyperparameter_results()
    export_backtest_results()
    export_predictions()
