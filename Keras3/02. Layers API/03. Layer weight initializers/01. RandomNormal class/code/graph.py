import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import RandomNormal
import seaborn as sns

# Parameters for the RandomNormal initializer
mean = 0.0
stddev = 0.05
seed = 42

# Create the RandomNormal initializer
initializer = RandomNormal(mean=mean, stddev=stddev, seed=seed)

# Generate a sample of 10,000 values
samples = initializer(shape=(10000,))

# Convert the samples to a NumPy array for easier plotting
samples = np.array(samples)

# Plot the histogram and KDE of the samples
plt.figure(figsize=(12, 6))

# Plot the histogram with custom bin size and transparency
plt.hist(samples, bins=50, density=True, alpha=0.5, color='b', edgecolor='black', label='RandomNormal Samples (Histogram)')

# Plot the KDE of the samples using seaborn
sns.kdeplot(samples, bw_adjust=0.5, color='red', label='KDE of RandomNormal Samples')

# Plot the theoretical normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5 * ((x - mean) / stddev) ** 2) / (stddev * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2, label='Theoretical Normal Distribution')

# Add labels, title, and legend
plt.title('RandomNormal Initialization Distribution with Histogram and KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Save the plot to a file in the current working directory
plt.savefig('random_normal_distribution.png')

# Show the plot
plt.show()
