import numpy as np
import matplotlib.pyplot as plt
import os

# Define the GELU function
def gelu(x):
    # Approximate the CDF of the standard normal distribution
    cdf_approx = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3 / (1 + 0.084204 * x**2))))
    return x * cdf_approx

# Generate a range of values
x = np.linspace(-5, 5, 400)
y = gelu(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{GELU}(x) = x \cdot \Phi(x)$', color='magenta')
plt.title(r'GELU Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{GELU}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'gelu_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
