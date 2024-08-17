import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Hard Sigmoid function
def hard_sigmoid(x):
    return np.clip((x + 3) / 6, 0, 1)

# Define the Hard SiLU function
def hard_silu(x):
    return x * hard_sigmoid(x)

# Generate a range of values
x = np.linspace(-5, 5, 400)
y = hard_silu(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{Hard SiLU}(x) = x \cdot \text{hard\_sigmoid}(x)$', color='blue')
plt.title(r'Hard SiLU Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{Hard SiLU}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'hard_silu_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
