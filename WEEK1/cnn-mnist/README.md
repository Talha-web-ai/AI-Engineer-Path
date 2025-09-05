# CNN on MNIST\n\nThis project implements a simple CNN to classify handwritten digits from the MNIST dataset.
# 🧠 CNN on MNIST – Day 2 (Week 1)

This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits from the **MNIST dataset** as part of my AI Engineer Path.

---

## 📌 Project Overview
- Dataset: **MNIST (60,000 training, 10,000 testing images of digits 0–9)**
- Model: **Simple CNN** with Conv → Pool → Conv → Pool → Dense layers
- Goal: Learn CNN basics (convolution, pooling, activation maps, architecture)

---

## 🏗️ Model Architecture
1. **Conv2D (32 filters, 3×3, ReLU)**
2. **MaxPooling2D (2×2)**
3. **Conv2D (64 filters, 3×3, ReLU)**
4. **MaxPooling2D (2×2)**
5. **Flatten**
6. **Dense (128, ReLU)**
7. **Dense (10, Softmax)**

---

## ⚙️ Requirements
Install dependencies:
```bash
pip install -r requirements.txt
