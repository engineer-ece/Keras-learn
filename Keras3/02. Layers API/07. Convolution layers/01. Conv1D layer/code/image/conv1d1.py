import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

# Create a dummy grayscale image (e.g., 64x64 pixels)
image = np.random.rand(64, 64)

# Flatten the image and reshape to fit Conv1D (e.g., treating rows as sequences)
image_rows = image  # Shape: (64, 64)

# Define a model with Conv1D
model = Sequential([
    Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(64, 64)),
    Flatten(),  # Flatten the output to feed into a Dense layer
    Dense(10, activation='softmax')  # Example output layer
])

# Print model summary
model.summary()

# Apply model
# Dummy input: For demonstration, use the image as input
image_expanded = np.expand_dims(image_rows, axis=0)  # Add batch dimension
output = model.predict(image_expanded)

# Plot original image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

# Plot output (for demonstration purposes, visualize output of the Dense layer)
plt.subplot(1, 2, 2)
plt.bar(range(10), output.flatten())  # Example output from Dense layer
plt.title('Model Output')

plt.show()