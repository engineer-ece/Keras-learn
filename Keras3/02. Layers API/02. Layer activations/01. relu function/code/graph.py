import numpy as np
import matplotlib.pyplot as plt
import os

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Generate a range of values
x = np.linspace(-10, 10, 400)
y = relu(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='ReLU(x) = max(0, x)', color='blue')
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'relu_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
