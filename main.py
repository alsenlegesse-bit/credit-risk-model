"""
Main script to run the credit risk model pipeline
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import DataProcessor
from train import ModelTrainer

def main():
    """Main pipeline"""
    print("=" * 60)
    print("CREDIT RISK MODEL PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Processing
    print("\n1. Loading and processing data...")
    processor = DataProcessor()
    
    # Load data (update path as needed)
    data_path = "data/train.csv"  # Update this path
    
    try:
        df = processor.load_data(data_path)
        print(f"   Data loaded: {df.shape}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        print("   Please ensure data/train.csv exists")
        return
    
    # Create features
    print("\n2. Creating RFM features...")
    rfm = processor.create_rfm_features()
    print(f"   RFM features created: {rfm.shape}")
    
    print("\n3. Creating target variable...")
    target, clusters = processor.create_target_variable(rfm)
    print(f"   Target created. High-risk customers: {target['is_high_risk'].sum()}")
    
    print("\n4. Creating customer features...")
    features = processor.create_customer_features()
    print(f"   Features created: {features.shape}")
    
    print("\n5. Preparing final dataset...")
    X, y = processor.prepare_data(features, target)
    print(f"   Final dataset: X={X.shape}, y={y.shape}")
    
    # Step 2: Model Training
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    trainer = ModelTrainer()
    
    print("\n6. Splitting data...")
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    print("\n7. Training models...")
    print("   This may take several minutes...")
    models = trainer.train_all_models()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    # Instructions
    print("\nNext steps:")
    print("1. Check MLflow UI: mlflow ui")
    print("2. Run API: uvicorn src.api.main:app --reload")
    print("3. Test API: http://localhost:8000/docs")
    print("4. Check saved models: best_model.pkl")

if __name__ == "__main__":
    main()
