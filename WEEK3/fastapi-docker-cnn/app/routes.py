from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from app.models import SimpleCNN

router = APIRouter()

# Load trained CNN model
model = SimpleCNN()
model.load_state_dict(torch.load("models/cnn_model.pth", map_location="cpu"))
model.eval()

# Define request schema
class PredictRequest(BaseModel):
    image_base64: str  # client will send image as base64 string

class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float

# Preprocessing transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure 1 channel (MNIST)
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        # Decode base64 image
        import base64
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # Transform
        tensor = transform(image).unsqueeze(0)  # shape [1,1,28,28]

        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

        return PredictResponse(
            predicted_class=predicted_class.item(),
            confidence=confidence.item()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
