# Import required libraries
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np

# Load and split the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape and normalize the data
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build the neural network model
model = Sequential([
    Input(shape=(784,)),
    Dense(units=128, activation='relu', name='layer1'),
    Dense(units=64, activation='relu', name='layer2'),
    Dense(units=10, activation='softmax', name='layer3')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
model.fit(X_train, y_train, epochs=10)

# Make predictions on test data
predictions = model.predict(X_test)
predict_label = np.argmax(predictions, axis=1)

# Print the first 25 predictions with actual labels
for i in range(25):
    print(f"{i+1}. Predicted: {predict_label[i]} - Actual: {y_test[i]}")

# Visualize the first 25 test images with predictions
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[i]}\nPred: {predict_label[i]}", fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/digit_predictions.png")
plt.show()

# Generate and save the confusion matrix
cm = confusion_matrix(y_test, predict_label)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.show()
