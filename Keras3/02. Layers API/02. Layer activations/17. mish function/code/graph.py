import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Mish function
def mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))

# Generate a range of values
x = np.linspace(-5, 5, 400)
y = mish(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))$', color='darkorange')
plt.title(r'Mish Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{Mish}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'mish_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
