import base64
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    # Check that the greeting mentions FastAPI
    assert "FastAPI" in body["message"]

def test_predict_sample_digit():
    # Read the sample digit image and encode to base64
    with open("data/sample_digit.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.post(
        "/predict",
        json={"image_base64": img_b64}
    )
    assert response.status_code == 200
    result = response.json()

    # Ensure result has required fields
    assert "predicted_class" in result
    assert "confidence" in result
    assert isinstance(result["predicted_class"], int)
    assert isinstance(result["confidence"], float)

    # For the saved digit (7), check it's classified correctly
    assert result["predicted_class"] == 7
    assert result["confidence"] > 0.9
