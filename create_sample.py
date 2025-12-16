import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Create directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Create sample dataset
np.random.seed(42)
n_samples = 5000

data = {
    'TransactionId': range(1000, 1000 + n_samples),
    'CustomerId': np.random.randint(100, 200, n_samples),
    'Amount': np.round(np.random.normal(150, 50, n_samples), 2),
    'TransactionStartTime': [
        (datetime(2024, 1, 1) + timedelta(
            days=np.random.randint(0, 90),
            hours=np.random.randint(0, 24)
        )).strftime('%Y-%m-%d %H:%M:%S')
        for _ in range(n_samples)
    ],
    'ProductCategory': np.random.choice(
        ['Electronics', 'Clothing', 'Groceries', 'Books'], 
        n_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    ),
    'ChannelId': np.random.choice(['Web', 'Android', 'IOS', 'Pay Later'], n_samples),
    'CountryCode': np.random.choice([1, 2, 3], n_samples),
    'FraudResult': np.random.choice([0, 1], n_samples, p=[0.96, 0.04])
}

df = pd.DataFrame(data)
df['Value'] = df['Amount'].abs()

# Save
df.to_csv('data/raw/sample_transactions.csv', index=False)
print(f"✅ Created sample dataset: {len(df)} rows")
print(f"Saved to: data/raw/sample_transactions.csv")
