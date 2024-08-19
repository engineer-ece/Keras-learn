import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import Constant
import seaborn as sns

# Parameters for the Constant initializer
value = 0.1  # The constant value used for initialization

# Create the Constant initializer
initializer = Constant(value=value)

# Generate a sample of 10,000 values
# For the Constant initializer, all values will be the same, so we will generate a matrix and flatten it
samples = initializer(shape=(100, 100)).numpy()  # Create a matrix to sample the constant weights

# Flatten the matrix to plot the values
samples_flattened = samples.flatten()

# Plot the histogram of the samples
plt.figure(figsize=(12, 6))

# Plot the histogram; since all values are constant, it will be a spike at the constant value
plt.hist(samples_flattened, bins=50, density=True, alpha=0.5, color='b', edgecolor='black', label='Constant Samples (Histogram)')

# Since all values are the same, KDE is not applicable
# Plot a vertical line at the constant value
plt.axvline(x=value, color='r', linestyle='--', label='Constant Value')

# Add labels, title, and legend
plt.title('Constant Initialization Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Save the plot to a file in the current working directory
plt.savefig('constant_distribution.png')

# Show the plot
plt.show()
