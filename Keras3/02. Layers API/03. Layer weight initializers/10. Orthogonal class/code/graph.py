import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import Orthogonal
import seaborn as sns

# Create the Orthogonal initializer
initializer = Orthogonal()

# Generate a sample of 10,000 values
samples = initializer(shape=(100, 100)).numpy()  # Create a matrix to sample the orthogonal weights

# Flatten the matrix to plot the values
samples_flattened = samples.flatten()

# Plot the histogram and KDE of the samples
plt.figure(figsize=(12, 6))

# Plot the histogram with custom bin size and transparency
plt.hist(samples_flattened, bins=50, density=True, alpha=0.5, color='b', edgecolor='black', label='Orthogonal Samples (Histogram)')

# Plot the KDE of the samples using seaborn
sns.kdeplot(samples_flattened, bw_adjust=0.5, color='red', label='KDE of Orthogonal Samples')

# Plot the theoretical normal distribution for comparison
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
stddev = np.sqrt(1 / samples.shape[0])  # Approximate standard deviation for normal distribution
p = np.exp(-0.5 * (x / stddev) ** 2) / (stddev * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2, label='Theoretical Normal Distribution')

# Add labels, title, and legend
plt.title('Orthogonal Initialization Distribution with Histogram and KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Save the plot to a file in the current working directory
plt.savefig('orthogonal_distribution.png')

# Show the plot
plt.show()
