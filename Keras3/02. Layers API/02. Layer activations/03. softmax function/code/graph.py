import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract np.max(x) for numerical stability
    return e_x / e_x.sum(axis=0)

# Generate a range of values
x = np.linspace(-2, 2, 400)
y = softmax(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{Softmax}(x) = \frac{e^x}{\sum e^x}$', color='purple')
plt.title(r'Softmax Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{Softmax}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'softmax_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
