"""
Simple Model Training Script
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def load_training_data():
    """Load train/test data"""
    print("Loading training data...")
    
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{model_name} Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics

def save_model(model, model_name):
    """Save model to file"""
    os.makedirs('models', exist_ok=True)
    filename = f'models/{model_name}.pkl'
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main():
    """Main training function"""
    print("=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Train models
    print("\n--- Training Models ---")
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate models
    print("\n--- Model Evaluation ---")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Select best model based on ROC AUC
    best_model = lr_model if lr_metrics['roc_auc'] > rf_metrics['roc_auc'] else rf_model
    best_name = "Logistic Regression" if lr_metrics['roc_auc'] > rf_metrics['roc_auc'] else "Random Forest"
    
    print(f"\nBest model: {best_name}")
    
    # Save best model
    save_model(best_model, "best_model")
    
    # Also save both models
    save_model(lr_model, "logistic_regression")
    save_model(rf_model, "random_forest")
    
    print("\nTraining completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
