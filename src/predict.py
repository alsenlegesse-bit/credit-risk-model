"""
Model prediction module
"""
import pandas as pd
import numpy as np
import joblib
from data_processing import DataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditRiskPredictor:
    """Make predictions with trained models"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.processor = DataProcessor()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_single(self, customer_data):
        """Predict for a single customer"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to DataFrame if needed
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Make prediction
        prediction = self.model.predict(customer_data)
        probability = self.model.predict_proba(customer_data)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'risk_level': 'HIGH' if prediction[0] == 1 else 'LOW'
        }
    
    def predict_batch(self, customers_data):
        """Predict for multiple customers"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Make predictions
        predictions = self.model.predict(customers_data)
        probabilities = self.model.predict_proba(customers_data)
        
        # Create results
        results = []
        for i in range(len(predictions)):
            results.append({
                'customer_id': i,
                'prediction': int(predictions[i]),
                'probability': float(probabilities[i][1]),
                'risk_level': 'HIGH' if predictions[i] == 1 else 'LOW'
            })
        
        return results
    
    def calculate_credit_score(self, probability):
        """Convert probability to credit score (300-850)"""
        # Simple linear scaling
        score = 300 + (550 * (1 - probability))
        return int(score)
    
    def recommend_loan_terms(self, probability, customer_features):
        """Recommend loan amount and duration based on risk"""
        # Base loan amount
        base_amount = 1000
        
        # Adjust based on probability
        if probability < 0.2:
            # Low risk
            amount = base_amount * 3
            duration = 24  # months
            interest_rate = 0.05
        elif probability < 0.5:
            # Medium risk
            amount = base_amount * 2
            duration = 12
            interest_rate = 0.08
        else:
            # High risk
            amount = base_amount
            duration = 6
            interest_rate = 0.12
        
        # Adjust based on customer's average transaction amount
        if 'avg_amount' in customer_features:
            avg_amount = customer_features['avg_amount']
            if avg_amount > 100:
                amount = min(amount * 2, 10000)  # Cap at 10,000
        
        return {
            'recommended_amount': float(amount),
            'recommended_duration_months': int(duration),
            'interest_rate': float(interest_rate),
            'monthly_payment': float(amount * interest_rate / 12 + amount / duration)
        }

if __name__ == "__main__":
    # Example usage
    predictor = CreditRiskPredictor("best_model.pkl")
    
    # Example customer data
    example_customer = {
        'total_amount': 5000,
        'avg_amount': 200,
        'transaction_count': 25,
        'unique_categories': 3,
        'fraud_rate': 0.0
    }
    
    result = predictor.predict_single(example_customer)
    print("Prediction Result:", result)
    
    credit_score = predictor.calculate_credit_score(result['probability'])
    print(f"Credit Score: {credit_score}")
    
    loan_terms = predictor.recommend_loan_terms(result['probability'], example_customer)
    print("Loan Terms:", loan_terms)
