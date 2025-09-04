🧠 ANN on Breast Cancer Dataset
📌 Project Overview

This project builds an Artificial Neural Network (ANN) to classify breast cancer tumors as benign (0) or malignant (1) using the Breast Cancer dataset from scikit-learn.
It’s a beginner-friendly but professional deep learning project, showcasing the full ML workflow from preprocessing to model evaluation.


📂 Project Structure
WEEK1/
 └── ANN_BreastCancer/
     ├── ann_breast_cancer.ipynb   # Main Jupyter Notebook
     ├── requirements.txt          # Dependencies
     └── README.md                 # Project documentation



🚀 Steps
Step 1: Imports

NumPy, Pandas → Data handling
Matplotlib, Seaborn → Visualization

Scikit-learn → Dataset loading, preprocessing, evaluation

TensorFlow/Keras → Building & training the ANN



Step 2: Load & Preprocess Data

Dataset: Breast Cancer dataset (30 features, binary labels)

Split: 80% Train, 20% Test

Preprocessing: StandardScaler → ensures features have mean=0, std=1



Step 3: Build ANN Model

Input Layer → 30 features

Hidden Layer 1 → 32 neurons, ReLU activation

Dropout → 20% (reduce overfitting)

Hidden Layer 2 → 16 neurons, ReLU activation

Output Layer → 1 neuron, Sigmoid activation (binary classification)

Compilation:

Optimizer: Adam

Loss: Binary Crossentropy

Metric: Accuracy



Step 4: Train Model

Epochs: 50

Batch size: 32

Validation split: 20%



Step 5: Evaluate Model

Accuracy Score

Classification Report (Precision, Recall, F1-score)

Confusion Matrix (visualized with heatmap)

Step 6: Training History Visualization

Training vs Validation Accuracy curve

Training vs Validation Loss curve

📊 Results

Achieved ~95-97% accuracy on test set

Balanced precision & recall

Clear separation in confusion matrix



🛠️ Installation & Setup
# Create conda environment
conda create -n week1 python=3.10 -y
conda activate week1

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook



📌 Learning Outcomes

✅ Understand ANN basics (layers, activation, optimizer, loss)
✅ Practice preprocessing pipeline (scaling, splitting)
✅ Build, train, and evaluate a neural network in Keras
✅ Visualize model performance (confusion matrix, accuracy/loss curves)
✅ Prepare a clean, portfolio-ready deep learning project

🔥 This project is part of my 10-week AI Engineer Roadmap, targeting top MNCs.