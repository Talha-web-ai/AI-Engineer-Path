# create_sample_digit.py
import numpy as np
from PIL import Image

# random digit-like image (28x28)
arr = (np.random.rand(28, 28) * 255).astype(np.uint8)
img = Image.fromarray(arr)
img.save("tests/sample_digit.png")
print("Sample image saved at tests/sample_digit.png")
