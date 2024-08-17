import numpy as np
import matplotlib.pyplot as plt
import os

# Define the ReLU6 function
def relu6(x):
    return np.minimum(np.maximum(x, 0), 6)

# Generate a range of values
x = np.linspace(-2, 8, 400)
y = relu6(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{ReLU6}(x) = \min(\max(x, 0), 6)$', color='red')
plt.title(r'ReLU6 Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{ReLU6}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'relu6_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
