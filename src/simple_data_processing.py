import pandas as pd
import numpy as np
import os

class SimpleFeatureEngineer:
    def __init__(self):
        self.df = None
        
    def load_data(self, filepath):
        """Load transaction data"""
        print(f"Loading data from {filepath}")
        if not os.path.exists(filepath):
            # Create sample data if file doesn't exist
            print("Creating sample data...")
            self.create_sample_data(filepath)
        
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} transactions")
        return self.df
    
    def create_sample_data(self, filepath):
        """Create sample transaction data"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        np.random.seed(42)
        n = 1000
        
        data = {
            'TransactionId': [f'TXN_{i:05d}' for i in range(n)],
            'CustomerId': [f'CUST_{np.random.randint(1, 101):03d}' for _ in range(n)],
            'Amount': np.random.uniform(-500, 500, n),
            'Value': np.random.uniform(1, 500, n),
            'TransactionStartTime': pd.date_range('2024-01-01', periods=n, freq='H'),
            'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n),
            'FraudResult': np.random.binomial(1, 0.05, n),
            'ChannelId': np.random.choice(['Web', 'Mobile', 'Android', 'iOS'], n),
            'CountryCode': np.random.choice(['ET', 'US', 'KE', 'NG'], n),
        }
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Created sample data with {len(df)} transactions")
        return df
    
    def create_features(self, df=None):
        """Create customer features"""
        if df is None:
            df = self.df
            
        print("Creating customer features...")
        
        # Ensure TransactionStartTime is datetime
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            
            # Calculate Recency (days since last transaction)
            latest_date = df['TransactionStartTime'].max()
            recency_df = df.groupby('CustomerId')['TransactionStartTime'].max()
            recency = (latest_date - recency_df).dt.days
            
        # Create aggregated features
        features = df.groupby('CustomerId').agg({
            'TransactionId': 'count',
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],
            'Value': ['sum', 'mean'],
            'FraudResult': 'sum'
        }).reset_index()
        
        # Flatten columns
        features.columns = ['CustomerId', 'TransactionCount', 'TotalAmount', 
                           'AvgAmount', 'StdAmount', 'MinAmount', 'MaxAmount',
                           'TotalValue', 'AvgValue', 'FraudCount']
        
        # Add Recency if calculated
        if 'TransactionStartTime' in df.columns:
            features = features.merge(recency.rename('Recency'), 
                                     left_on='CustomerId', 
                                     right_index=True)
        
        # Calculate Frequency and Monetary
        features['Frequency'] = features['TransactionCount']
        features['Monetary'] = features['TotalAmount'].abs()
        
        # Calculate ratios
        features['FraudRatio'] = features['FraudCount'] / features['TransactionCount'].replace(0, 1)
        
        # Fill NaN values
        features = features.fillna(0)
        
        print(f"Created features for {len(features)} customers")
        return features
    
    def save_features(self, features, output_path='data/processed/customer_features.csv'):
        """Save features to CSV"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features.to_csv(output_path, index=False)
        print(f"Features saved to {output_path}")

def main():
    """Main function"""
    engineer = SimpleFeatureEngineer()
    
    # Load or create data
    df = engineer.load_data('data/raw/transactions.csv')
    
    # Create features
    features = engineer.create_features(df)
    
    # Save features
    engineer.save_features(features)
    
    return features

if __name__ == "__main__":
    main()
