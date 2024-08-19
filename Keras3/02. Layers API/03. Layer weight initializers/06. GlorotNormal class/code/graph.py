import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import GlorotNormal
import seaborn as sns

# Parameters for the GlorotNormal initializer
mean = 0.0
stddev = 0.05  # Approximate value for plotting purposes; actual stddev is dynamically calculated

# Create the GlorotNormal initializer
initializer = GlorotNormal()

# Generate a sample of 10,000 values
samples = initializer(shape=(10000,)).numpy()  # Convert to a NumPy array for easier plotting

# Plot the histogram and KDE of the samples
plt.figure(figsize=(12, 6))

# Plot the histogram with custom bin size and transparency
plt.hist(samples, bins=50, density=True, alpha=0.5, color='b', edgecolor='black', label='GlorotNormal Samples (Histogram)')

# Plot the KDE of the samples using seaborn
sns.kdeplot(samples, bw_adjust=0.5, color='red', label='KDE of GlorotNormal Samples')

# Plot the theoretical normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
# Theoretical standard deviation is approximately sqrt(2 / (fan_in + fan_out))
# Here we use stddev=0.05 for illustrative purposes, actual stddev varies
p = np.exp(-0.5 * (x / stddev) ** 2) / (stddev * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2, label='Theoretical Normal Distribution')

# Add labels, title, and legend
plt.title('GlorotNormal Initialization Distribution with Histogram and KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Save the plot to a file in the current working directory
plt.savefig('glorot_normal_distribution.png')

# Show the plot
plt.show()
