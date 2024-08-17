import numpy as np
import matplotlib.pyplot as plt
import os

# Define the Softplus function
def softplus(x):
    return np.log1p(np.exp(x))  # log1p(x) is more accurate for small x values

# Generate a range of values
x = np.linspace(-10, 10, 400)
y = softplus(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label=r'$\text{Softplus}(x) = \log(1 + e^x)$', color='orange')
plt.title(r'Softplus Function')
plt.xlabel(r'$x$')
plt.ylabel(r'$\text{Softplus}(x)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()

# Get the current working directory
current_folder = os.getcwd()

# Save the plot as an image file in the current folder
file_path = os.path.join(current_folder, 'softplus_function.png')
plt.savefig(file_path)

print(f"Plot saved as {file_path}")

# Optionally show the plot
# plt.show()
