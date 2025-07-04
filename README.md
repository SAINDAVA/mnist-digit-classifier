# MNIST Digit Classification using TensorFlow

This project classifies handwritten digits (0-9) from the MNIST dataset using a Neural Network built with TensorFlow and Keras.

## 📦 Dataset
- MNIST dataset (60,000 training samples, 10,000 test samples)
- Each image is 28x28 grayscale

## 🧠 Model Architecture
- **Input Layer**: Flatten 28x28 into 784
- **Dense Layer 1**: 128 units, ReLU
- **Dense Layer 2**: 64 units, ReLU
- **Output Layer**: 10 units, Softmax

## 🧪 Training
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 10

## 📊 Results
- Training accuracy: ~99%
- Confusion Matrix shows performance over all digit classes

## 📷 Output Samples
Visualized test images with:
- True label
- Predicted label

![Sample Predictions](digit_predictions.png)

## 🧮 Confusion Matrix
Confusion matrix visualizes prediction errors per class.

![Confusion Matrix](confusion_matrix.png)

## 🚀 How to Run
```bash
pip install tensorflow numpy matplotlib
python mnist_classifier.py
