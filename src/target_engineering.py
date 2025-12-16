"""
Target Variable Engineering for Credit Risk Model
Task 4: Create proxy target variable using RFM clustering
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

def load_features(filepath='data/processed/customer_features.csv'):
    """Load processed features"""
    print(f"📂 Loading features from {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df)} customers with {df.shape[1]} features")
    return df

def create_target_variable(features_df):
    """
    Create is_high_risk target using K-Means clustering on RFM features
    """
    print("\n🎯 Creating target variable using RFM clustering...")
    
    # Required RFM features
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    
    # Check if RFM features exist
    missing_cols = [col for col in rfm_cols if col not in features_df.columns]
    if missing_cols:
        print(f"Warning: Missing RFM columns: {missing_cols}")
        print("   Creating dummy target (20% high risk)...")
        features_df['is_high_risk'] = 0
        # Randomly assign 20% as high risk
        high_risk_idx = features_df.sample(frac=0.2, random_state=42).index
        features_df.loc[high_risk_idx, 'is_high_risk'] = 1
        return features_df
    
    # Prepare RFM data
    rfm_data = features_df[rfm_cols].copy()
    
    # Handle NaN/inf values
    rfm_data = rfm_data.replace([np.inf, -np.inf], np.nan)
    rfm_data = rfm_data.fillna(rfm_data.median())
    
    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    # Apply K-Means clustering (3 clusters as required)
    print("   Applying K-Means clustering (n_clusters=3)...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(rfm_scaled)
    
    # Analyze clusters
    features_df['rfm_cluster'] = cluster_labels
    cluster_stats = features_df.groupby('rfm_cluster')[rfm_cols].mean()
    
    print("\n📊 Cluster Statistics:")
    print(cluster_stats)
    
    # Identify high-risk cluster (highest Recency = most inactive)
    high_risk_cluster = cluster_stats['Recency'].idxmax()
    print(f"\n🔍 High-risk cluster identified: Cluster {high_risk_cluster}")
    print("   Characteristics: High Recency (inactive), Low Frequency, Low Monetary")
    
    # Create binary target
    features_df['is_high_risk'] = (features_df['rfm_cluster'] == high_risk_cluster).astype(int)
    
    # Remove temporary cluster column
    features_df = features_df.drop('rfm_cluster', axis=1)
    
    # Display target distribution
    print(f"\n🎯 Target Distribution:")
    n_high_risk = features_df['is_high_risk'].sum()
    n_total = len(features_df)
    print(f"   High-risk customers: {n_high_risk:,} ({n_high_risk/n_total:.1%})")
    print(f"   Low-risk customers: {n_total - n_high_risk:,} ({1 - n_high_risk/n_total:.1%})")
    
    return features_df

def prepare_train_test_split(features_with_target, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    print("\n📊 Preparing train/test split...")
    
    # Separate features and target
    X = features_with_target.drop(['CustomerId', 'is_high_risk'], axis=1, errors='ignore')
    y = features_with_target['is_high_risk']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]:,} samples")
    print(f"   Test set: {X_test.shape[0]:,} samples")
    print(f"   Feature count: {X_train.shape[1]}")
    print(f"   Target distribution (train): {y_train.mean():.3f}")
    print(f"   Target distribution (test): {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test

def save_split_data(X_train, X_test, y_train, y_test, output_dir='data/processed'):
    """
    Save train/test data to CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False, header=['is_high_risk'])
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False, header=['is_high_risk'])
    
    print(f"\n💾 Data saved to {output_dir}/")
    print(f"   X_train.csv: {X_train.shape}")
    print(f"   X_test.csv: {X_test.shape}")
    print(f"   y_train.csv: {y_train.shape}")
    print(f"   y_test.csv: {y_test.shape}")

def main():
    """Main function for target engineering pipeline"""
    print("=" * 50)
    print("TARGET VARIABLE ENGINEERING PIPELINE")
    print("=" * 50)
    
    # Load features
    features = load_features()
    if features is None:
        print("Cannot proceed without features. Run data_processing.py first.")
        return
    
    # Create target variable
    features_with_target = create_target_variable(features)
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(features_with_target)
    
    # Save processed data
    save_split_data(X_train, X_test, y_train, y_test)
    
    # Save full dataset with target
    features_with_target.to_csv('data/processed/features_with_target.csv', index=False)
    
    print("\nTarget engineering completed!")
    print("=" * 50)
    
    return features_with_target, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()
