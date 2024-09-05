import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers, initializers

# Custom Regularizer class
class CustomRegularizer(regularizers.Regularizer):
    def __init__(self, l=0.01):
        self.l = l

    def __call__(self, x):
        return self.l * np.sum(np.square(x))

# Parameters for RandomNormal initializer
mean = 0.0
stddev = 0.05
seed = 42

# Create the RandomNormal initializer
initializer = initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)

# Generate sample weights
weights_no_reg = initializer(shape=(10000,))
weights_with_reg = initializer(shape=(10000,))

# Apply custom regularizer to the weights (simulating the effect)
regularizer = CustomRegularizer(l=0.01)
weights_with_reg -= regularizer(weights_with_reg)

# Convert weights to NumPy arrays for easier plotting
weights_no_reg = np.array(weights_no_reg)
weights_with_reg = np.array(weights_with_reg)

# Theoretical PDF calculations
def normal_pdf(x, mean, stddev):
    return (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

def truncated_normal_pdf(x, mean, stddev, cutoff):
    pdf = normal_pdf(x, mean, stddev)
    pdf[x < -cutoff] = 0
    pdf[x > cutoff] = 0
    return pdf / np.sum(pdf)

# Define theoretical parameters
cutoff = 2 * stddev  # Example cutoff for truncated normal distribution
x_vals = np.linspace(-0.2, 0.2, 1000)

# Calculate theoretical PDFs
pdf_no_reg = normal_pdf(x_vals, mean, stddev)
pdf_with_reg = truncated_normal_pdf(x_vals, mean, stddev, cutoff)

# Plot the histogram and KDE of the weights
plt.figure(figsize=(14, 7))

# Plot the histogram with custom bin size and transparency for weights without regularization
plt.hist(weights_no_reg, bins=50, density=True, alpha=0.5, color='blue', edgecolor='black', label='Weights without Regularization')

# Plot the KDE of the weights without regularization using seaborn
sns.kdeplot(weights_no_reg, bw_adjust=0.5, color='blue', label='KDE of Weights without Regularization')

# Plot the histogram with custom bin size and transparency for weights with regularization
plt.hist(weights_with_reg, bins=50, density=True, alpha=0.5, color='red', edgecolor='black', label='Weights with Regularization')

# Plot the KDE of the weights with regularization using seaborn
sns.kdeplot(weights_with_reg, bw_adjust=0.5, color='red', label='KDE of Weights with Regularization')

# Plot theoretical PDFs
plt.plot(x_vals, pdf_no_reg, 'b--', label=r'Theoretical PDF without Reg: $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{x^2}{2\sigma^2}}$')
plt.plot(x_vals, pdf_with_reg, 'g*', label=r'Theoretical PDF with Reg: Truncated Normal with $\sigma=0.05$')

# Add labels, title, and legend
plt.title('Weight Distributions with and without Regularization')
plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend(loc='upper left')

# Save the plot to a file in the current working directory
plt.savefig('weight_distributions_with_regularization_theoretical.png')

# Show the plot
plt.show()
