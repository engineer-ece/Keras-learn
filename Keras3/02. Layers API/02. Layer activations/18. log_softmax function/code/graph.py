import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Log Softmax function
def log_softmax(x):
    e_x = np.exp(x - np.max(x))  # Stability trick to avoid overflow
    return np.log(e_x / e_x.sum(axis=0))

# Generate a range of values
x = np.linspace(-5, 5, 400)
y = log_softmax(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{LogSoftmax}(x_i) = x_i - \log\left(\sum_j e^{x_j}\right)$', color='brown')
plt.title(r'Log Softmax Function')
plt.xlabel(r'$x_i$')
plt.ylabel(r'$\text{LogSoftmax}(x_i)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'log_softmax_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
