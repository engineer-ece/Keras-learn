import numpy as np
import matplotlib.pyplot as plt
import os

# Define the SiLU function
def silu(x):
    return x * (1 / (1 + np.exp(-x)))

# Generate a range of values
x = np.linspace(-5, 5, 400)
y = silu(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{SiLU}(x) = x \cdot \frac{1}{1 + e^{-x}}$', color='purple')
plt.title(r'SiLU (Swish) Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{SiLU}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'silu_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
