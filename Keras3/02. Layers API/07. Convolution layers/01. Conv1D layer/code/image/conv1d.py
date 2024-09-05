# Key Adjustments
# Batch Dimension Removal: The conv1d_output has shape (1, 28, 26). We need to remove the batch dimension using conv1d_output[0] to get the shape (28, 26).

# Visualization: plt.imshow requires a 2D array for grayscale images or 2D arrays for each channel. Ensure that conv1d_output_reshaped is a 2D array.

# By making these adjustments, the code will properly handle the dimensions and allow correct visualization of the original image and the Conv1D output.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv1D, Input
from tensorflow.keras.models import Model

# Load MNIST dataset
(train_images, _), (_, _) = mnist.load_data()

# Select a single image from the dataset
image = train_images[0]  # Shape: (28, 28)

# Normalize and preprocess the image
image_normalized = image.astype('float32') / 255.0
image_reshaped = image_normalized.reshape((28, 28))  # Shape: (28, 28)

# Reshape for Conv1D input: (height, width)
image_reshaped_for_conv1d = image_reshaped.reshape((28, 28))  # Shape: (28, 28)

# Define a model with Conv1D
input_layer = Input(shape=(28, 28))  # Input shape: (rows, columns)
conv1d_layer = Conv1D(filters=1, kernel_size=3, activation='relu', padding='valid')(input_layer)

# Create a model to apply Conv1D
model = Model(inputs=input_layer, outputs=conv1d_layer)

# Apply the model to the image data
image_expanded = np.expand_dims(image_reshaped_for_conv1d, axis=0)  # Add batch dimension
conv1d_output = model.predict(image_expanded)

# Extract the output from Conv1D and reshape for visualization
# Conv1D output shape is (1, 28, 26) where 26 is the width after applying kernel size 3
conv1d_output_reshaped = conv1d_output[0]  # Shape: (28, 26), remove batch dimension

# Plot the original image and Conv1D output
plt.figure(figsize=(12, 6))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(image_reshaped, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot the Conv1D output
plt.subplot(1, 2, 2)
plt.imshow(conv1d_output_reshaped, cmap='viridis')
plt.title('After Conv1D')
plt.axis('off')

plt.tight_layout()
plt.show()
