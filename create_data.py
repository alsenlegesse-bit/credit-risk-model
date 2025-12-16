import pandas as pd
import numpy as np
import os

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Create sample transactions
np.random.seed(42)
n = 5000
data = {
    'TransactionId': [f'TXN_{i}' for i in range(n)],
    'CustomerId': [f'CUST_{np.random.randint(1, 101)}' for _ in range(n)],
    'Amount': np.random.uniform(-500, 500, n),
    'TransactionStartTime': pd.date_range('2024-01-01', periods=n, freq='H'),
    'ProductCategory': np.random.choice(['Electronics', 'Clothing', 'Food'], n),
    'FraudResult': np.random.binomial(1, 0.05, n)
}
df = pd.DataFrame(data)
df['Value'] = df['Amount'].abs()
df.to_csv('data/raw/transactions.csv', index=False)
print(f"Created {len(df)} transactions")
