Day 3 – CNN on CIFAR-10
📌 Overview

This project implements a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.
We integrate Batch Normalization, Dropout, and Data Augmentation to improve generalization.

The model is trained to classify 10 object categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

📖 Concepts Covered

Batch Normalization → Speeds up training and stabilizes gradients.

Dropout → Prevents overfitting by randomly deactivating neurons.

Data Augmentation → Improves robustness with rotation, shifts, and flips.

Deep CNN with multiple convolutional blocks.



🧑‍💻 Project Structure
WEEK1/cnn-cifar10/
│── CNN_CIFAR10.ipynb   # Notebook with training, evaluation, and visualization
│── requirements.txt    # Required dependencies
│── README.md           # Project documentation


⚙️ Requirements
Install dependencies:

pip install -r requirements.txt



🚀 Training the Model
Run the notebook:

jupyter notebook CNN_CIFAR10.ipynb



Training details:

Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Epochs: 20
Batch Size: 64
Augmentation: Rotation, horizontal flip, width/height shifts


📊 Results

Achieves ~75–80% accuracy on CIFAR-10 test set (depending on epochs).

Loss and accuracy curves are plotted for better understanding.

Sample predictions are visualized with their labels.



🔮 Future Improvements

Implement ResNet or Inception architectures.
Try learning rate schedulers.
Train longer with GPU acceleration for better performance.



✅ Deliverable

This is the Day 3 project in the AI Engineer Path (WEEK1).
It demonstrates modern CNN techniques for image classification.