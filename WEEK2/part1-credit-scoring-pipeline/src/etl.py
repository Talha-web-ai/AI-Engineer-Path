import pandas as pd
import os
import logging

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(
    filename='etl.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_etl():
    logging.info("🔹 Running ETL pipeline...")

    # ----------------------------
    # Load raw dataset
    # ----------------------------
    try:
        data = pd.read_csv("data/credit_data.csv") # 👈 raw file
        logging.info(f"Loaded raw dataset with shape {data.shape}")
    except Exception as e:
        logging.error(f"Error loading raw dataset: {e}")
        raise

    # ----------------------------
    # Basic cleaning (example)
    # ----------------------------
    # Drop duplicates
    data = data.drop_duplicates()

    # Fill missing values (numeric → median, categorical → mode)
    for col in data.columns:
        if data[col].dtype in ["int64", "float64"]:
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])

    logging.info("Applied basic cleaning (duplicates removed, missing values filled)")

    # ----------------------------
    # Save processed dataset
    # ----------------------------
    os.makedirs("data", exist_ok=True)
    processed_path = "data/credit_data.csv"
    data.to_csv(processed_path, index=False)

    logging.info(f"✅ ETL completed. Processed dataset saved at {processed_path}")
    print("✅ ETL completed. Processed dataset ready for training!")

if __name__ == "__main__":
    run_etl()
