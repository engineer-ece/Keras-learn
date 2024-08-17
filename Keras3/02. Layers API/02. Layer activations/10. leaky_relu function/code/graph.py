import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Leaky ReLU function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Generate a range of values
x = np.linspace(-5, 5, 400)
y = leaky_relu(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{Leaky ReLU}(x) = x \text{ if } x > 0 \text{ else } \alpha x$', color='orange')
plt.title(r'Leaky ReLU Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{Leaky ReLU}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'leaky_relu_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
