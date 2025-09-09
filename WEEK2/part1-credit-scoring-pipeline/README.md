# Credit Scoring Pipeline (Day 1 Project)

A simple ML pipeline for predicting credit default using Random Forest.  
Includes feature scaling, model training, metrics tracking, and logging.

---

## 📁 Project Structure

part1-credit-scoring-pipeline/
├── data/
│ └── credit_data.csv # Input dataset
├── models/
│ ├── credit_model.pkl # Trained model
│ └── scaler.pkl # Feature scaler
├── train.py # Main training script
├── metrics.json # Model performance metrics
├── train.log # Log file for training process
├── README.md
└── requirements.txt


## ⚙️ Setup

1. **Clone the repo**
```bash
git clone <repo_url>
cd part1-credit-scoring-pipeline
Create a virtual environment (optional but recommended)


python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies

pip install -r requirements.txt

🚀 Run Training


python train.py

What happens:

Loads credit_data.csv
Splits into train/test sets
Scales features
Trains Random Forest
Logs all steps to train.log
Saves metrics to metrics.json
Saves model & scaler in models/

📊 Output
Metrics: metrics.json

json
Copy code
{
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.90,
    "f1_score": 0.91
}
Logs: train.log

Model & Scaler: models/credit_model.pkl, models/scaler.pkl

