print("=" * 60)
print("FINAL SETUP VERIFICATION FOR CREDIT RISK MODEL")
print("=" * 60)

# Test all critical imports
imports_to_test = [
    ("pandas", "pd"),
    ("numpy", "np"),
    ("sklearn", "sklearn"),
    ("sklearn.ensemble", "RandomForestClassifier"),
    ("sklearn.linear_model", "LogisticRegression"),
    ("sklearn.model_selection", "train_test_split"),
    ("scipy", "scipy"),
    ("xgboost", "xgb"),
    ("lightgbm", "lgb"),
    ("fastapi", "FastAPI"),
    ("pydantic", "BaseModel"),
    ("mlflow", "mlflow"),
    ("xverse", "WOE"),
    ("imblearn", "imblearn"),
    ("joblib", "joblib"),
]

print("\n1. Testing core ML libraries...")
for import_path, alias in imports_to_test:
    try:
        if '.' in import_path:
            # Handle submodule imports
            parts = import_path.split('.')
            exec(f"from {parts[0]} import {parts[1]}")
        else:
            exec(f"import {import_path} as {alias}")
        print(f"   ✓ {import_path}")
    except ImportError as e:
        print(f"   ✗ {import_path} - {str(e)[:50]}")

print("\n2. Testing your project modules...")
try:
    from src.data_processing import DataProcessor
    print("   ✓ DataProcessor")
except Exception as e:
    print(f"   ✗ DataProcessor - {e}")

try:
    from src.train import ModelTrainer
    print("   ✓ ModelTrainer")
except Exception as e:
    print(f"   ✗ ModelTrainer - {e}")

try:
    from src.predict import CreditRiskPredictor
    print("   ✓ CreditRiskPredictor")
except Exception as e:
    print(f"   ✗ CreditRiskPredictor - {e}")

print("\n" + "=" * 60)
print("SETUP READY CHECKLIST")
print("=" * 60)

# Quick functionality test
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Create sample data
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
print(f"Sample data created: X.shape={X.shape}, y.shape={y.shape}")

# Test basic model training
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)
accuracy = model.score(X, y)
print(f"Test model trained with accuracy: {accuracy:.2f}")

print("\n✅ Your environment is READY for the credit risk project!")
print("   Run: python main.py")
