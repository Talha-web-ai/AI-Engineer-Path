# Create a dummy dataset to test
import pandas as pd
import os

os.makedirs('../data', exist_ok=True)

data = pd.DataFrame({
    'age': [25, 45, 35, 50, 23],
    'income': [50000, 80000, 60000, 90000, 40000],
    'loan_amount': [20000, 30000, 25000, 40000, 15000],
    'default': [0, 1, 0, 1, 0]
})

data.to_csv('../data/credit_data.csv', index=False)
print("Dummy dataset created at data/credit_data.csv")
