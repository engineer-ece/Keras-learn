import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import Identity
import seaborn as sns

# Parameters for the Identity initializer
shape = (10, 10)  # Identity initializer is typically used for square matrices

# Create the Identity initializer
initializer = Identity(gain=1.0)

# Generate a sample of values (Note: Identity initializer is for square matrices)
samples = initializer(shape).numpy().flatten()  # Flatten to create a 1D array for easier plotting

# Plot the histogram and KDE of the samples
plt.figure(figsize=(12, 6))

# Plot the histogram with custom bin size and transparency
plt.hist(samples, bins=50, density=True, alpha=0.5, color='b', edgecolor='black', label='Identity Samples (Histogram)')

# Plot the KDE of the samples using seaborn
sns.kdeplot(samples, bw_adjust=0.5, color='red', label='KDE of Identity Samples')

# Plot the theoretical distribution (only valid for identity matrix entries, which are 0 or 1)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.zeros_like(x)
p[x == 1] = 1.0  # Identity matrix has values 1 on the diagonal
plt.plot(x, p, 'k', linewidth=2, label='Theoretical Distribution')

# Add labels, title, and legend
plt.title('Identity Initialization Distribution with Histogram and KDE')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend()

# Save the plot to a file in the current working directory
plt.savefig('identity_distribution.png')

# Show the plot
plt.show()
