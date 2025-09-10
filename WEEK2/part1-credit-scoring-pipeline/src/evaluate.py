import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import logging

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(
    filename='evaluate.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def evaluate():
    # ----------------------------
    # Load dataset
    # ----------------------------
    try:
        data = pd.read_csv('data/credit_data.csv')
        logging.info(f'Dataset loaded successfully with shape {data.shape}')
    except Exception as e:
        logging.error(f'Error loading dataset: {e}')
        raise

    X = data.drop("default", axis=1)
    y = data["default"]

    # ----------------------------
    # Load pipeline model
    # ----------------------------
    try:
        pipeline = joblib.load("models/credit_model.pkl")
        logging.info("Pipeline model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

    # ----------------------------
    # Predict
    # ----------------------------
    y_pred = pipeline.predict(X)

    # ----------------------------
    # Evaluate
    # ----------------------------
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }

    logging.info(f"Evaluation Metrics: {metrics}")

    # ----------------------------
    # Save metrics
    # ----------------------------
    with open("evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info("Evaluation metrics saved to evaluation_metrics.json")

    print("✅ Evaluation complete. Metrics saved!")

if __name__ == "__main__":
    evaluate()
