from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
import joblib
from src.utils.helpers import load_model

app = FastAPI(title="FinSight API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler at startup
MODEL_PATH = Path('models/best_model.pkl')
SCALER_PATH = Path('models/best_scaler.pkl')
model = load_model(MODEL_PATH)
scaler = load_model(SCALER_PATH)


class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    regime: str


@app.get("/")
def root():
    return {
        "message": "FinSight API - Market Regime Classification",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Convert input features to DataFrame
        input_df = pd.DataFrame([request.features])
        # Scale features
        input_scaled = scaler.transform(input_df)
        # Predict
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        regime = "bullish" if pred == 1 else "bearish"
        return PredictionResponse(
            prediction=int(pred),
            probability=float(prob),
            regime=regime
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Dashboard Data Endpoints
@app.get("/dashboard/summary")
def get_summary():
    """Get profitability summary for dashboard."""
    try:
        summary_path = Path('powerbi_summary.csv')
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="Summary data not found. Run generate_powerbi_report.py first.")
        df = pd.read_csv(summary_path)
        return df.to_dict(orient='records')[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/predictions")
def get_predictions(limit: Optional[int] = 500):
    """Get predictions data for dashboard charts."""
    try:
        predictions_path = Path('powerbi_predictions.csv')
        if not predictions_path.exists():
            raise HTTPException(status_code=404, detail="Predictions data not found.")
        df = pd.read_csv(predictions_path)
        # Return last N records for performance
        df = df.tail(limit) if limit else df
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/monthly")
def get_monthly():
    """Get monthly performance data for dashboard."""
    try:
        monthly_path = Path('powerbi_monthly.csv')
        if not monthly_path.exists():
            raise HTTPException(status_code=404, detail="Monthly data not found.")
        df = pd.read_csv(monthly_path)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/hyperparameters")
def get_hyperparameters():
    """Get hyperparameter tuning results."""
    try:
        hp_path = Path('models/hyperparameter_results.csv')
        if not hp_path.exists():
            raise HTTPException(status_code=404, detail="Hyperparameter results not found.")
        df = pd.read_csv(hp_path)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
