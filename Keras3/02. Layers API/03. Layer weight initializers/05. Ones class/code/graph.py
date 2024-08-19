import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import Ones

# Create the Ones initializer
initializer = Ones()

# Generate a sample of 10,000 values
samples = initializer(shape=(10000,)).numpy()  # Convert to a NumPy array for easier manipulation

# Plot the histogram of the samples
plt.figure(figsize=(12, 6))

# Plot the histogram with custom bin size and transparency
plt.hist(samples, bins=50, density=True, alpha=0.5, color='b', edgecolor='black', label='Ones Samples (Histogram)')

# Since all values are one, there's no need for KDE or theoretical distribution plots

# Add labels, title, and legend
plt.title('Ones Initialization Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Save the plot to a file in the current working directory
plt.savefig('ones_distribution.png')

# Show the plot
plt.show()
