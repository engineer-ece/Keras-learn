import numpy as np
import matplotlib.pyplot as plt
import os

# Define the ELU function
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Generate a range of values
x = np.linspace(-5, 5, 400)
y = elu(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{ELU}(x) = \lambda x \text{ if } x > 0 \text{ else } \alpha (e^x - 1)$', color='blue')
plt.title(r'Exponential Linear Unit (ELU) Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{ELU}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'elu_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
