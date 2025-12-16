from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Try to load model
try:
    model = joblib.load('models/best_model.pkl')
    model_loaded = True
except:
    model = None
    model_loaded = False

@app.get("/")
def root():
    return {"message": "Credit Risk API", "model_loaded": model_loaded}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/predict_sample")
def predict_sample():
    """Make a sample prediction"""
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    # Create sample input
    sample_input = np.random.randn(1, 13)  # 13 features based on our data
    
    # Predict
    prediction = model.predict(sample_input)
    probability = model.predict_proba(sample_input)[0, 1]
    
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability),
        "risk_level": "High" if prediction[0] == 1 else "Low"
    }
