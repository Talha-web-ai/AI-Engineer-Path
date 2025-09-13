🧠 FastAPI + Docker: CNN Digit Classifier
📌 Project Overview

This project demonstrates how to serve a Deep Learning model (CNN) using FastAPI and package it into a Docker container for reproducible deployments.
The model is trained on the MNIST dataset (handwritten digits 0-9) and exposed via a REST API.

✅ Day 1 (Week 3):

Built and trained a CNN model for digit classification.

Saved the trained model for serving.

✅ Day 2 (Week 3):

Implemented FastAPI endpoints (/ and /predict).

Wrote tests with pytest to validate API responses.

Successfully containerized the app using Docker.

🚀 Features

🧩 CNN Model – trained on MNIST

⚡ FastAPI – high-performance REST API

🧪 Pytest – automated testing

🐳 Dockerized – portable & reproducible deployment

✅ CI-Ready – project structure supports pipelines

📂 Project Structure
fastapi-docker-cnn/
│── app/
│   ├── main.py           # FastAPI application
│   ├── model.py          # CNN model loading + prediction
│   └── utils.py          # Helper functions
│── tests/
│   └── test_api.py       # Pytest API tests
│── Dockerfile            # Docker build file
│── requirements.txt      # Dependencies
│── train.py              # Model training script
│── README.md             # Project documentation

🛠️ Setup & Installation
1️⃣ Clone the repo
git clone https://github.com/talha-web-ai/AI-Engineer-Path.git
cd AI-Engineer-Path/WEEK3/fastapi-docker-cnn

2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3️⃣ Run FastAPI locally
uvicorn app.main:app --reload


API will be available at: http://127.0.0.1:8000

🐳 Run with Docker
# Build image
docker build -t fastapi-cnn .

# Run container
docker run -p 8000:8000 fastapi-cnn

🧪 Run Tests
PYTHONPATH=. pytest -v


✅ All tests should pass

📡 API Endpoints
GET /

Returns a welcome message.

Response:

{ "message": "Welcome to CNN Digit Classifier API 🚀" }

POST /predict

Send an image (28x28 grayscale digit) for prediction.

Sample Response:

{
  "digit": 7,
  "confidence": 0.9876
}