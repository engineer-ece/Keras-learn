import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import HeUniform
import seaborn as sns

# Parameters for the HeUniform initializer
# The limit is determined by the formula sqrt(6 / fan_in)
limit = np.sqrt(6.0 / 1.0)  # Using fan_in = 1 for visualization; adjust as necessary

# Create the HeUniform initializer
initializer = HeUniform()

# Generate a sample of 10,000 values
samples = initializer(shape=(10000,)).numpy()  # Convert to a NumPy array for easier plotting

# Plot the histogram and KDE of the samples
plt.figure(figsize=(12, 6))

# Plot the histogram with custom bin size and transparency
plt.hist(samples, bins=50, density=True, alpha=0.5, color='b', edgecolor='black', label='HeUniform Samples (Histogram)')

# Plot the KDE of the samples using seaborn
sns.kdeplot(samples, bw_adjust=0.5, color='red', label='KDE of HeUniform Samples')

# Plot the theoretical uniform distribution
xmin, xmax = plt.xlim()
x = np.linspace(-limit, limit, 100)
p = np.ones_like(x) / (2 * limit)  # Uniform distribution formula in the range [-limit, limit]
plt.plot(x, p, 'k', linewidth=2, label='Theoretical Uniform Distribution')

# Add labels, title, and legend
plt.title('HeUniform Initialization Distribution with Histogram and KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Save the plot to a file in the current working directory
plt.savefig('he_uniform_distribution.png')

# Show the plot
plt.show()
