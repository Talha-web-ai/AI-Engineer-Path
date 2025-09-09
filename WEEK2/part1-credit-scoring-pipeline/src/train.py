import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import logging
import os

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
X = data.drop("default", axis=1)
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logging.info(f'Data split into train ({X_train.shape}) and test ({X_test.shape}) sets')

# ----------------------------
# Scale features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logging.info('Feature scaling applied')

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
logging.info('RandomForestClassifier trained successfully')

# ----------------------------
# Evaluate model
# ----------------------------
y_pred = model.predict(X_test_scaled)

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
# Optional: Save model & scaler for future use
# ----------------------------
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/credit_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
logging.info('Model and scaler saved to models/')
