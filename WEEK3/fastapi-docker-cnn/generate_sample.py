import base64
from torchvision import datasets, transforms
from PIL import Image
import requests

# -------------------
# Step 1: Load MNIST test sample
# -------------------
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Pick first sample (digit 7 usually)
image, label = test_dataset[0]

# Convert tensor -> PIL
pil_img = transforms.ToPILImage()(image)

# Save to file for reference
out_path = "data/sample_digit.png"
pil_img.save(out_path)
print(f"✅ Saved sample digit (label={label}) to {out_path}")

# -------------------
# Step 2: Convert to base64
# -------------------
with open(out_path, "rb") as f:
    img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()

# -------------------
# Step 3: Call FastAPI
# -------------------
url = "http://127.0.0.1:8000/predict"
payload = {"image_base64": img_b64}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    print("✅ Prediction:", response.json())
except Exception as e:
    print("❌ Error calling FastAPI:", e)
