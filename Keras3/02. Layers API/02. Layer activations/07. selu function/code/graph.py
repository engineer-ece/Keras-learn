import numpy as np
import matplotlib.pyplot as plt
import os

# Define the SELU function
def selu(x, alpha=1.67326, lambda_=1.0507):
    return lambda_ * np.where(x > 0, x, alpha * np.exp(x) - alpha)

# Generate a range of values
x = np.linspace(-5, 5, 400)
y = selu(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{SELU}(x) = \lambda, x \text{ if } x > 0 \text{ else } \alpha e^x - \alpha$', color='green')
plt.title(r'Scaled Exponential Linear Unit (SELU) Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{SELU}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'selu_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
