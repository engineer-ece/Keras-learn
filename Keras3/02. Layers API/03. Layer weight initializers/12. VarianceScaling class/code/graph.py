import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import VarianceScaling
import seaborn as sns

# Parameters for the VarianceScaling initializer
scale = 1.0       # Scaling factor (default is 1.0)
mode = 'fan_avg'  # Mode for scaling: 'fan_in', 'fan_out', 'fan_avg'
seed = 42         # Seed for reproducibility

# Create the VarianceScaling initializer
initializer = VarianceScaling(scale=scale, mode=mode, seed=seed)

# Generate a sample of 10,000 values
samples = initializer(shape=(10000,)).numpy()  # Convert to a NumPy array for easier plotting

# Plot the histogram and KDE of the samples
plt.figure(figsize=(12, 6))

# Plot the histogram with custom bin size and transparency
plt.hist(samples, bins=50, density=True, alpha=0.5, color='b', edgecolor='black', label='VarianceScaling Samples (Histogram)')

# Plot the KDE of the samples using seaborn
sns.kdeplot(samples, bw_adjust=0.5, color='red', label='KDE of VarianceScaling Samples')

# Plot the theoretical distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
# Theoretical distribution for VarianceScaling is roughly Gaussian with scaling
stddev = np.sqrt(scale)  # Standard deviation is sqrt(scale)
p = np.exp(-0.5 * (x / stddev) ** 2) / (stddev * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2, label='Theoretical Gaussian Distribution')

# Add labels, title, and legend
plt.title('VarianceScaling Initialization Distribution with Histogram and KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Save the plot to a file in the current working directory
plt.savefig('variance_scaling_distribution.png')

# Show the plot
plt.show()
