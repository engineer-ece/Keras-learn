import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Hard Sigmoid function
def hard_sigmoid(x):
    return np.clip((x + 2.5) / 5, 0, 1)

# Generate a range of values
x = np.linspace(-5, 5, 400)
y = hard_sigmoid(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{Hard Sigmoid}(x) = \max(0, \min(1, \frac{x + 2.5}{5}))$', color='purple')
plt.title(r'Hard Sigmoid Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{Hard Sigmoid}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'hard_sigmoid_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
