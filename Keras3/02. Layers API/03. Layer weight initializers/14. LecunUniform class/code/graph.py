import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import LecunUniform
import seaborn as sns

# Parameters for the LeCunUniform initializer
minval = -0.05  # Approximate minval for plotting purposes; actual minval is dynamically calculated
maxval = 0.05   # Approximate maxval for plotting purposes; actual maxval is dynamically calculated

# Create the LeCunUniform initializer
initializer = LecunUniform(seed=42)

# Generate a sample of 10,000 values
samples = initializer(shape=(10000,)).numpy()  # Convert to a NumPy array for easier plotting

# Plot the histogram and KDE of the samples
plt.figure(figsize=(12, 6))

# Plot the histogram with custom bin size and transparency
plt.hist(samples, bins=50, density=True, alpha=0.5, color='b', edgecolor='black', label='LeCunUniform Samples (Histogram)')

# Plot the KDE of the samples using seaborn
sns.kdeplot(samples, bw_adjust=0.5, color='red', label='KDE of LeCunUniform Samples')

# Plot the theoretical uniform distribution
xmin, xmax = plt.xlim()
x = np.linspace(minval, maxval, 100)
p = np.ones_like(x) / (maxval - minval)  # Uniform distribution formula
plt.plot(x, p, 'k', linewidth=2, label='Theoretical Uniform Distribution')

# Add labels, title, and legend
plt.title('LeCun Uniform Initialization Distribution with Histogram and KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Save the plot to a file in the current working directory
plt.savefig('lecun_uniform_distribution.png')

# Show the plot
plt.show()
