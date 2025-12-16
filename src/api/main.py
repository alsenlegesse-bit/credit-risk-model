"""
FastAPI application for credit risk prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import numpy as np
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import CreditRiskPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using ML models",
    version="1.0.0"
)

# Load model
try:
    predictor = CreditRiskPredictor("best_model.pkl")
    logger.info("Model loaded successfully")
except:
    logger.warning("Model not found. Running without model.")

# Pydantic models
class CustomerFeatures(BaseModel):
    """Customer features for prediction"""
    total_amount: float
    avg_amount: float
    std_amount: float
    transaction_count: int
    total_value: float
    unique_categories: int
    unique_channels: int
    avg_hour: float
    fraud_rate: float
    
    class Config:
        schema_extra = {
            "example": {
                "total_amount": 5000.0,
                "avg_amount": 200.0,
                "std_amount": 50.0,
                "transaction_count": 25,
                "total_value": 5500.0,
                "unique_categories": 3,
                "unique_channels": 2,
                "avg_hour": 14.5,
                "fraud_rate": 0.0
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response"""
    customer_id: Optional[str]
    prediction: int
    probability: float
    risk_level: str
    credit_score: int
    recommended_amount: float
    recommended_duration_months: int
    interest_rate: float
    monthly_payment: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str

# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/predict",
            "/predict-batch",
            "/docs"
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: CustomerFeatures, customer_id: Optional[str] = None):
    """Predict credit risk for a single customer"""
    try:
        # Convert features to dict
        features_dict = features.dict()
        
        # Make prediction
        prediction = predictor.predict_single(features_dict)
        
        # Calculate credit score
        credit_score = predictor.calculate_credit_score(prediction['probability'])
        
        # Get loan terms
        loan_terms = predictor.recommend_loan_terms(prediction['probability'], features_dict)
        
        # Prepare response
        response = {
            "customer_id": customer_id,
            "prediction": prediction['prediction'],
            "probability": prediction['probability'],
            "risk_level": prediction['risk_level'],
            "credit_score": credit_score,
            **loan_terms
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(features_list: List[CustomerFeatures]):
    """Predict credit risk for multiple customers"""
    try:
        results = []
        
        for i, features in enumerate(features_list):
            features_dict = features.dict()
            
            # Make prediction
            prediction = predictor.predict_single(features_dict)
            
            # Calculate credit score
            credit_score = predictor.calculate_credit_score(prediction['probability'])
            
            # Get loan terms
            loan_terms = predictor.recommend_loan_terms(prediction['probability'], features_dict)
            
            # Prepare response
            response = {
                "customer_id": f"customer_{i}",
                "prediction": prediction['prediction'],
                "probability": prediction['probability'],
                "risk_level": prediction['risk_level'],
                "credit_score": credit_score,
                **loan_terms
            }
            
            results.append(response)
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    fraud_rate: float
    
    class Config:
        schema_extra = {
            "example": {
                "total_amount": 5000.0,
                "avg_amount": 200.0,
                "std_amount": 50.0,
                "transaction_count": 25,
                "total_value": 5500.0,
                "unique_categories": 3,
                "unique_channels": 2,
                "avg_hour": 14.5,
                "fraud_rate": 0.0
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response"""
    customer_id: Optional[str]
    prediction: int
    probability: float
    risk_level: str
    credit_score: int
    recommended_amount: float
    recommended_duration_months: int
    interest_rate: float
    monthly_payment: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str

# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/predict",
            "/predict-batch",
            "/docs"
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: CustomerFeatures, customer_id: Optional[str] = None):
    """Predict credit risk for a single customer"""
    try:
        # Convert features to dict
        features_dict = features.dict()
        
        # Make prediction
        prediction = predictor.predict_single(features_dict)
        
        # Calculate credit score
        credit_score = predictor.calculate_credit_score(prediction['probability'])
        
        # Get loan terms
        loan_terms = predictor.recommend_loan_terms(prediction['probability'], features_dict)
        
        # Prepare response
        response = {
            "customer_id": customer_id,
            "prediction": prediction['prediction'],
            "probability": prediction['probability'],
            "risk_level": prediction['risk_level'],
            "credit_score": credit_score,
            **loan_terms
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(features_list: List[CustomerFeatures]):
    """Predict credit risk for multiple customers"""
    try:
        results = []
        
        for i, features in enumerate(features_list):
            features_dict = features.dict()
            
            # Make prediction
            prediction = predictor.predict_single(features_dict)
            
            # Calculate credit score
            credit_score = predictor.calculate_credit_score(prediction['probability'])
            
            # Get loan terms
            loan_terms = predictor.recommend_loan_terms(prediction['probability'], features_dict)
            
            # Prepare response
            response = {
                "customer_id": f"customer_{i}",
                "prediction": prediction['prediction'],
                "probability": prediction['probability'],
                "risk_level": prediction['risk_level'],
                "credit_score": credit_score,
                **loan_terms
            }
            
            results.append(response)
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
