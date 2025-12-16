"""
Model training with MLflow tracking
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import xgboost as xgb
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Complete model training pipeline"""
    
    def __init__(self, experiment_name="credit_risk_model"):
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {self.X_train.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Train class balance: {self.y_train.mean():.3%}")
        logger.info(f"Test class balance: {self.y_test.mean():.3%}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        return metrics
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        run_name = f"logistic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Model
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            
            # Hyperparameters
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear']
            }
            
            # Grid search
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=0
            )
            
            # Train
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Log model
            signature = infer_signature(self.X_train, best_model.predict(self.X_train))
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
            
            # Save locally
            joblib.dump(best_model, "logistic_regression_model.pkl")
            mlflow.log_artifact("logistic_regression_model.pkl")
            
            logger.info(f"Logistic Regression - Best params: {grid_search.best_params_}")
            logger.info(f"Logistic Regression - ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return best_model, metrics
    
    def train_random_forest(self):
        """Train Random Forest model"""
        run_name = f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Model
            model = RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # Hyperparameters
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
            
            # Grid search
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=0
            )
            
            # Train
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Log model
            signature = infer_signature(self.X_train, best_model.predict(self.X_train))
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
            
            logger.info(f"Random Forest - Best params: {grid_search.best_params_}")
            logger.info(f"Random Forest - ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return best_model, metrics
    
    def train_xgboost(self):
        """Train XGBoost model"""
        run_name = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1])
            
            # Model
            model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1
            )
            
            # Hyperparameters
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
            
            # Grid search
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='roc_auc',
                n_jobs=-1, verbose=0
            )
            
            # Train
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log to MLflow
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            
            # Log model
            signature = infer_signature(self.X_train, best_model.predict(self.X_train))
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
            
            logger.info(f"XGBoost - Best params: {grid_search.best_params_}")
            logger.info(f"XGBoost - ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return best_model, metrics
    
    def train_all_models(self):
        """Train all models and compare"""
        logger.info("=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60)
        
        models_to_train = {
            'logistic_regression': self.train_logistic_regression,
            'random_forest': self.train_random_forest,
            'xgboost': self.train_xgboost
        }
        
        for model_name, train_func in models_to_train.items():
            try:
                logger.info(f"\nTraining {model_name.replace('_', ' ').title()}...")
                model, metrics = train_func()
                
                self.models[model_name] = {
                    'model': model,
                    'metrics': metrics
                }
                
                # Update best model
                if metrics['roc_auc'] > self.best_score:
                    self.best_score = metrics['roc_auc']
                    self.best_model = model
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Print comparison
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        for model_name, model_info in self.models.items():
            metrics = model_info['metrics']
            logger.info(f"\n{model_name.upper().replace('_', ' ')}:")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        logger.info(f"\nBest Model ROC-AUC: {self.best_score:.4f}")
        
        # Save best model
        if self.best_model:
            joblib.dump(self.best_model, "best_model.pkl")
            logger.info("Best model saved as 'best_model.pkl'")
        
        return self.models

if __name__ == "__main__":
    # Example usage
    from data_processing import DataProcessor
    
    # Load and prepare data
    processor = DataProcessor()
    df = processor.load_data("data/train.csv")
    rfm = processor.create_rfm_features()
    target, _ = processor.create_target_variable(rfm)
    features = processor.create_customer_features()
    X, y = processor.prepare_data(features, target)
    
    # Train models
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    models = trainer.train_all_models()
    
    print("\nTraining complete!")
