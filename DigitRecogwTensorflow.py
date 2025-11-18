import cv2 as cv # OpenCV lib for computer vision tasks (image processing)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the MNIST dataset - 70,000 handwritten digits (0-9)
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# Normalize the pixel values from 0-255 to 0-1 to help the model train better
x_train = tf.keras.utils.normalize(x_train, axis=-1)
x_test = tf.keras.utils.normalize(x_test, axis=-1)

# Create a Sequential model - a linear stack of layers
model = tf.keras.models.Sequential()

# Flatten layer to convert 28x28 images into 1D arrays of 784 pixels
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

# Add a Dense (fully connected) layer with 128 neurons and ReLU activation
# ReLU helps the model learn complex patterns by only keeping positive values
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))

# Add another Dense layer with 64 neurons and ReLU activation
# This creates a deeper network that can learn more complex features
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))

# Add the output layer with 10 neurons (one for each digit 0-9) and softmax activation
# Softmax converts the outputs into probabilities that sum to 1
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compile the model by specifying:
# - optimizer: 'adam' - algorithm that adjusts weights to minimize loss
# - loss: 'sparse_categorical_crossentropy' - measures how wrong the predictions are
# - metrics: ['accuracy'] - tracks what percentage of predictions are correct
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 5 epochs (passes through the entire training dataset)
model.fit(x_train, y_train, epochs=5)

loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy:', accuracy)
print('Loss:', loss)

model.save('digits.model.keras')

for x in range(1, 6):  # 1 to 5
    img = cv.imread(f"{x}.jpg") # Load image

    # Convert to grayscale and resize to 28x28 (same as MNIST)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (28, 28))

    # Invert (MNIST has white digits on black background)
    img = np.invert(img)
    img = img / 255.0  # Normalize to 0-1

    # Reshape for model prediction
    img = np.array([img])

    # Predict
    prediction = model.predict(img)
    print(f'Prediction: {np.argmax(prediction)}')

    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.title(f'Prediction: {np.argmax(prediction)}')
    plt.show()
