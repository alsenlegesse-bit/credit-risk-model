"""
Complete data processing pipeline for credit risk model
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Main data processing class"""
    
    def __init__(self):
        self.df = None
        self.features_df = None
        self.target_df = None
        
    def load_data(self, filepath):
        """Load transaction data"""
        try:
            self.df = pd.read_csv(filepath)
            logger.info(f"Data loaded. Shape: {self.df.shape}")
            
            # Convert datetime
            self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
            
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_rfm_features(self):
        """Create Recency, Frequency, Monetary features"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Set snapshot date
        snapshot_date = self.df['TransactionStartTime'].max() + timedelta(days=1)
        
        # Calculate RFM
        rfm = self.df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': lambda x: x[x > 0].sum()
        }).reset_index()
        
        rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        
        # Handle zeros
        rfm['Monetary'] = rfm['Monetary'].fillna(0)
        
        # Log transform
        rfm['Recency_log'] = np.log1p(rfm['Recency'])
        rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
        rfm['Monetary_log'] = np.log1p(rfm['Monetary'] + 1)
        
        logger.info(f"RFM features created. Shape: {rfm.shape}")
        return rfm
    
    def create_target_variable(self, rfm_df, n_clusters=3):
        """Create target variable using RFM clustering"""
        # Prepare features
        clustering_features = rfm_df[['Recency_log', 'Frequency_log', 'Monetary_log']].copy()
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(clustering_features)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Assign clusters
        rfm_df['cluster'] = clusters
        
        # Analyze clusters
        cluster_stats = rfm_df.groupby('cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerId': 'count'
        })
        
        # Calculate risk score
        cluster_stats['risk_score'] = (
            cluster_stats['Recency'].rank(ascending=False) +
            cluster_stats['Frequency'].rank(ascending=True) +
            cluster_stats['Monetary'].rank(ascending=True)
        )
        
        # Identify high-risk cluster
        high_risk_cluster = cluster_stats['risk_score'].idxmax()
        
        # Create binary target
        rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
        
        self.target_df = rfm_df[['CustomerId', 'is_high_risk']]
        
        logger.info(f"High-risk customers: {self.target_df['is_high_risk'].sum()}")
        logger.info(f"Percentage: {self.target_df['is_high_risk'].mean():.2%}")
        
        return self.target_df, cluster_stats
    
    def create_customer_features(self):
        """Create comprehensive customer features"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Aggregate features per customer
        agg_features = self.df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'count', 'min', 'max'],
            'Value': ['sum', 'mean', 'std'],
            'TransactionId': 'nunique',
            'ProductCategory': 'nunique',
            'ChannelId': 'nunique',
            'ProviderId': 'nunique',
            'CountryCode': 'nunique'
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                               for col in agg_features.columns.values]
        
        # Rename columns
        agg_features = agg_features.rename(columns={
            'CustomerId_': 'CustomerId',
            'Amount_sum': 'total_amount',
            'Amount_mean': 'avg_amount',
            'Amount_std': 'std_amount',
            'Amount_count': 'transaction_count',
            'Amount_min': 'min_amount',
            'Amount_max': 'max_amount',
            'Value_sum': 'total_value',
            'Value_mean': 'avg_value',
            'Value_std': 'std_value',
            'TransactionId_nunique': 'unique_transactions',
            'ProductCategory_nunique': 'unique_categories',
            'ChannelId_nunique': 'unique_channels',
            'ProviderId_nunique': 'unique_providers',
            'CountryCode_nunique': 'unique_countries'
        })
        
        # Calculate ratios
        agg_features['amount_variability'] = agg_features['std_amount'] / (agg_features['avg_amount'] + 1e-6)
        agg_features['value_to_amount_ratio'] = agg_features['total_value'] / (agg_features['total_amount'] + 1e-6)
        
        # Temporal features
        self.df['hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['day_of_week'] = self.df['TransactionStartTime'].dt.dayofweek
        self.df['month'] = self.df['TransactionStartTime'].dt.month
        
        temporal_features = self.df.groupby('CustomerId').agg({
            'hour': ['mean', 'std'],
            'day_of_week': ['mean', 'std'],
            'month': 'nunique'
        }).reset_index()
        
        temporal_features.columns = ['CustomerId', 'avg_hour', 'std_hour', 
                                     'avg_day_of_week', 'std_day_of_week', 'unique_months']
        
        # Merge all features
        features = pd.merge(agg_features, temporal_features, on='CustomerId', how='left')
        
        # Behavioral features
        fraud_features = self.df.groupby('CustomerId')['FraudResult'].agg(['sum', 'mean']).reset_index()
        fraud_features.columns = ['CustomerId', 'fraud_count', 'fraud_rate']
        
        features = pd.merge(features, fraud_features, on='CustomerId', how='left')
        
        self.features_df = features
        logger.info(f"Features created. Shape: {self.features_df.shape}")
        
        return self.features_df
    
    def prepare_data(self, features_df, target_df):
        """Merge features and target"""
        data = pd.merge(features_df, target_df, on='CustomerId', how='inner')
        
        # Separate features and target
        X = data.drop(['CustomerId', 'is_high_risk'], axis=1, errors='ignore')
        y = data['is_high_risk']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        logger.info(f"Final dataset: {X.shape}, Target: {y.shape}")
        logger.info(f"Class distribution: {y.mean():.2%}")
        
        return X, y

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    df = processor.load_data("data/train.csv")
    rfm = processor.create_rfm_features()
    target, clusters = processor.create_target_variable(rfm)
    features = processor.create_customer_features()
    X, y = processor.prepare_data(features, target)
    
    print("Data processing complete!")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
