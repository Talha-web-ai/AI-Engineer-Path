import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import json
import logging
import os
import joblib

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(
    filename='train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ----------------------------
# Load dataset
# ----------------------------
try:
    data = pd.read_csv('data/credit_data.csv')
    logging.info(f'Dataset loaded successfully with shape {data.shape}')
except Exception as e:
    logging.error(f'Error loading dataset: {e}')
    raise

# ----------------------------
# Split features & target
# ----------------------------
X = data.drop("default", axis=1)   # 👈 using "default" as target
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logging.info(f'Data split into train ({X_train.shape}) and test ({X_test.shape}) sets')

# ----------------------------
# Build pipeline (scaler + model)
# ----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ----------------------------
# Train model
# ----------------------------
pipeline.fit(X_train, y_train)
logging.info('RandomForestClassifier trained successfully (via pipeline)')

# ----------------------------
# Evaluate model
# ----------------------------
y_pred = pipeline.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

logging.info(f'Metrics: {metrics}')

# ----------------------------
# Save metrics to JSON
# ----------------------------
metrics_path = 'metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
logging.info(f'Metrics saved to {metrics_path}')

# ----------------------------
# Save pipeline model
# ----------------------------
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/credit_model.pkl')
logging.info('Pipeline (scaler + model) saved to models/credit_model.pkl')
