print("="*50)
print("TASK 2: EXPLORATORY DATA ANALYSIS")
print("="*50)

import pandas as pd
import numpy as np

print("\n1. Creating sample dataset...")

# Create sample data
np.random.seed(42)
data = {
    'TransactionId': range(1, 1001),
    'CustomerId': np.random.randint(100, 120, 1000),
    'Amount': np.round(np.random.normal(150, 50, 1000), 2),
    'ProductCategory': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'ChannelId': np.random.choice(['Web', 'Android', 'IOS'], 1000),
    'FraudResult': np.random.choice([0, 1], 1000, p=[0.97, 0.03])
}

df = pd.DataFrame(data)
df['Value'] = df['Amount'].abs()

# Save to CSV
import os
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/sample_data.csv', index=False)

print(f"Created dataset: {len(df)} rows, {len(df.columns)} columns")

print("\n2. Basic Analysis:")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values: {df.isnull().sum().sum()}")

print("\n3. Summary Statistics:")
print(df[['Amount', 'Value']].describe())

print("\n4. Value Counts:")
print("\nProduct Categories:")
print(df['ProductCategory'].value_counts())
print("\nChannels:")
print(df['ChannelId'].value_counts())
print("\nFraud Results:")
print(df['FraudResult'].value_counts())

print("\n5. Key Insights:")
print(f"- Dataset size: {len(df)} transactions")
print(f"- Unique customers: {df['CustomerId'].nunique()}")
print(f"- Average transaction: ${df['Amount'].mean():.2f}")
print(f"- Fraud rate: {df['FraudResult'].mean():.1%}")
print(f"- Most common channel: {df['ChannelId'].mode()[0]}")

print("\n" + "="*50)
print("✅ TASK 2 EDA COMPLETED!")
print("="*50)
