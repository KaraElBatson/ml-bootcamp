"""
FastAPI web application for serving ML model predictions
"""
from typing import Dict, List, Optional, Tuple
import pickle
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from scripts.config import MODELS_DIR, MODEL_FILENAME
from scripts.inference import load_model_pipeline
from scripts.model_utils import load_model


# Initialize FastAPI app
app = FastAPI(title="Credit Scoring ML API", description="API for credit risk prediction")

# Model placeholder
model_pipeline = None

# Define input model for API
class CreditFeatures(BaseModel):
    age: int = Field(25, description="Age of applicant")
    income: float = Field(35000, description="Annual income")
    employment_length: float = Field(5, description="Years in current employment")
    home_ownership: str = Field("MORTGAGE", description="Home ownership status")
    purpose: str = Field("debt_consolidation", description="Purpose of loan")
    credit_amount: float = Field(10000, description="Requested credit amount")
    interest_rate: float = Field(10.5, description="Interest rate of loan")
    percent_income: float = Field(20, description="Credit amount as % of income")
    has_defaults: int = Field(0, description="Has previous defaults")
    years_at_residence: float = Field(3, description="Years at current residence")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Prediction (0=rejected, 1=approved)")
    probability: float = Field(..., description="Probability of credit approval")
    message: str = Field(..., description="Human-readable result message")


class BatchRequest(BaseModel):
    applicants: List[CreditFeatures] = Field(..., description="List of applicants for batch prediction")


@app.on_event("startup")
async def load_model():
    global model_pipeline
    
    try:
        model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model_pipeline = load_model_pipeline()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
async def root():
    return {"message": "Credit Scoring ML API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: CreditFeatures):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to dictionary
        data = features.dict()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model_pipeline.predict(df)[0]
        
        # Get probability
        if hasattr(model_pipeline.named_steps["model"], "predict_proba"):
            probability = model_pipeline.predict_proba(df)[0, 1]  # Probability of class 1
        else:
            probability = 0.0
            
        # Prepare response
        status = "approved" if prediction == 1 else "rejected"
        message = f"Credit application {status}"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert list of Pydantic models to DataFrame
        data_list = [features.dict() for features in request.applicants]
        df = pd.DataFrame(data_list)
        
        # Make predictions
        predictions = model_pipeline.predict(df)
        
        # Get probabilities
        if hasattr(model_pipeline.named_steps["model"], "predict_proba"):
            probabilities = model_pipeline.predict_proba(df)[:, 1]  # Probability of class 1
        else:
            probabilities = np.zeros(len(predictions))
        
        # Prepare response
        results = []
        for i, prediction in enumerate(predictions):
            status = "approved" if prediction == 1 else "rejected"
            results.append({
                "applicant_index": i,
                "prediction": int(prediction),
                "probability": float(probabilities[i]),
                "message": f"Credit application {status}"
            })
        
        return {"results": results, "total_applicants": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    from scripts.config import API_CONFIG
    uvicorn.run(
        "app:app", 
        host=API_CONFIG["host"], 
        port=API_CONFIG["port"], 
        reload=API_CONFIG["debug"]
    )