from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from pathlib import Path
import joblib
from src.utils.helpers import load_model

app = FastAPI(title="FinSight API", version="1.0.0")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
