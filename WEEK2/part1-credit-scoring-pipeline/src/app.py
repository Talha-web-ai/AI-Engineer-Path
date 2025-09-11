from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "✅ Credit Scoring API is running!"}


# -------------------------
# Load the trained model
# -------------------------
MODEL_PATH = os.path.join("models", "credit_model.pkl")
model = joblib.load(MODEL_PATH)

# -------------------------
# Prediction endpoint
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert JSON to DataFrame (assuming features are passed in correctly)
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------------
# Health check
# -------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "API is running 🚀"})

if __name__ == "__main__":
    app.run(debug=True)
