import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import RandomNormal

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

# Plot the histogram of the samples
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, color='b')

# Plot the theoretical normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5*((x - mean)/stddev)**2) / (stddev * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2)

# Add labels and title
plt.title('RandomNormal Initialization Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)

# Show the plot
plt.show()
