"""
Unit tests for data processing module
"""
import pytest
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_processor_initialization():
    """Test DataProcessor initialization"""
    processor = DataProcessor()
    assert processor.df is None
    assert processor.features_df is None
    assert processor.target_df is None
    print("✓ test_data_processor_initialization passed")

def test_create_rfm_features():
    """Test RFM feature creation"""
    # Create sample data
    data = {
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionStartTime': pd.date_range('2023-01-01', periods=5),
        'TransactionId': [100, 101, 102, 103, 104],
        'Amount': [100, 200, 50, 150, -20]
    }
    df = pd.DataFrame(data)
    
    processor = DataProcessor()
    processor.df = df
    
    rfm = processor.create_rfm_features()
    
    # Check columns
    expected_columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 
                       'Recency_log', 'Frequency_log', 'Monetary_log']
    assert all(col in rfm.columns for col in expected_columns)
    
    # Check shape
    assert rfm.shape[0] == 3  # 3 unique customers
    
    print("✓ test_create_rfm_features passed")

def test_create_target_variable():
    """Test target variable creation"""
    # Create sample RFM data
    rfm_data = {
        'CustomerId': [1, 2, 3, 4, 5],
        'Recency': [10, 20, 30, 40, 50],
        'Frequency': [5, 4, 3, 2, 1],
        'Monetary': [1000, 800, 600, 400, 200],
        'Recency_log': np.log1p([10, 20, 30, 40, 50]),
        'Frequency_log': np.log1p([5, 4, 3, 2, 1]),
        'Monetary_log': np.log1p([1000, 800, 600, 400, 200])
    }
    rfm_df = pd.DataFrame(rfm_data)
    
    processor = DataProcessor()
    target, clusters = processor.create_target_variable(rfm_df, n_clusters=2)
    
    # Check columns
    assert 'CustomerId' in target.columns
    assert 'is_high_risk' in target.columns
    
    # Check binary values
    assert set(target['is_high_risk'].unique()).issubset({0, 1})
    
    print("✓ test_create_target_variable passed")

def test_create_customer_features():
    """Test customer feature creation"""
    # Create sample transaction data
    data = {
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionStartTime': pd.date_range('2023-01-01', periods=5),
        'Amount': [100, 200, 50, 150, 75],
        'Value': [100, 200, 50, 150, 75],
        'TransactionId': [100, 101, 102, 103, 104],
        'ProductCategory': ['A', 'B', 'A', 'C', 'B'],
        'ChannelId': ['Web', 'Mobile', 'Web', 'Mobile', 'Web'],
        'ProviderId': ['P1', 'P2', 'P1', 'P1', 'P2'],
        'CountryCode': [1, 1, 2, 2, 1],
        'FraudResult': [0, 0, 1, 0, 0]
    }
    df = pd.DataFrame(data)
    
    processor = DataProcessor()
    processor.df = df
    
    features = processor.create_customer_features()
    
    # Check key columns exist
    assert 'CustomerId' in features.columns
    assert 'total_amount' in features.columns
    assert 'transaction_count' in features.columns
    assert 'unique_categories' in features.columns
    
    # Check shape
    assert features.shape[0] == 3  # 3 unique customers
    
    print("✓ test_create_customer_features passed")

def test_prepare_data():
    """Test data preparation"""
    # Create sample features
    features_data = {
        'CustomerId': [1, 2, 3],
        'total_amount': [1000, 2000, 3000],
        'transaction_count': [10, 20, 30]
    }
    features_df = pd.DataFrame(features_data)
    
    # Create sample target
    target_data = {
        'CustomerId': [1, 2, 3],
        'is_high_risk': [0, 1, 0]
    }
    target_df = pd.DataFrame(target_data)
    
    processor = DataProcessor()
    X, y = processor.prepare_data(features_df, target_df)
    
    # Check shapes
    assert X.shape[0] == 3
    assert len(y) == 3
    
    # Check that CustomerId and target are removed from X
    assert 'CustomerId' not in X.columns
    assert 'is_high_risk' not in X.columns
    
    print("✓ test_prepare_data passed")

if __name__ == "__main__":
    # Run all tests
    test_data_processor_initialization()
    test_create_rfm_features()
    test_create_target_variable()
    test_create_customer_features()
    test_prepare_data()
    print("\n✅ All tests passed!")
