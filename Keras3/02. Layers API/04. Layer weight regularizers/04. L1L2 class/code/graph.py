import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers, initializers, models, layers

# Custom Regularizer class for L1L2 Regularization
class CustomL1L2Regularizer(regularizers.Regularizer):
    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x):
        return self.l1 * np.sum(np.abs(x)) + self.l2 * np.sum(np.square(x))

# Function to create a model and extract weights
def create_model_with_l1_l2_regularization(l1_value, l2_value):
    model = models.Sequential([
        layers.Dense(64, input_shape=(100,), kernel_regularizer=regularizers.l1_l2(l1=l1_value, l2=l2_value), 
                     kernel_initializer=initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Parameters for RandomNormal initializer
mean = 0.0
stddev = 0.05
seed = 42
l1_value = 0.01  # L1 regularization strength
l2_value = 0.01  # L2 regularization strength

# Create models and extract weights
model_custom_reg = create_model_with_l1_l2_regularization(l1_value, l2_value)
model_builtin_reg = create_model_with_l1_l2_regularization(l1_value, l2_value)

# Generate random input data
X = np.random.randn(1000, 100)
model_custom_reg.fit(X, np.random.randn(1000, 1), epochs=1, verbose=0)
model_builtin_reg.fit(X, np.random.randn(1000, 1), epochs=1, verbose=0)

# Extract weights
weights_no_reg = model_custom_reg.layers[0].get_weights()[0].flatten()
weights_with_reg = model_builtin_reg.layers[0].get_weights()[0].flatten()

# Convert weights to NumPy arrays for easier plotting
weights_no_reg = np.array(weights_no_reg)
weights_with_reg = np.array(weights_with_reg)

# Theoretical PDF calculations
def normal_pdf(x, mean, stddev):
    return (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

def laplace_pdf(x, mean, b):
    return (1 / (2 * b)) * np.exp(-np.abs(x - mean) / b)

def combined_pdf(x, mean, stddev, b):
    # Combine Gaussian and Laplace PDFs for visualization purposes
    pdf_gaussian = normal_pdf(x, mean, stddev)
    pdf_laplace = laplace_pdf(x, mean, b)
    return 0.5 * (pdf_gaussian + pdf_laplace)

# Define theoretical parameters
b = stddev / np.sqrt(2)  # Scale parameter for Laplace distribution (approximates L1 regularization effect)
x_vals = np.linspace(-0.2, 0.2, 1000)

# Calculate theoretical PDFs
pdf_no_reg = normal_pdf(x_vals, mean, stddev)
pdf_with_reg = combined_pdf(x_vals, mean, stddev, b)  # Combined effect of L1 and L2 regularization

# Plot the histogram and KDE of the weights
plt.figure(figsize=(14, 7))

# Plot the histogram with custom bin size and transparency for weights without regularization
plt.hist(weights_no_reg, bins=50, density=True, alpha=0.5, color='blue', edgecolor='black', label='Weights without Regularization')

# Plot the KDE of the weights without regularization using seaborn
sns.kdeplot(weights_no_reg, bw_adjust=0.5, color='blue', label='KDE of Weights without Regularization')

# Plot the histogram with custom bin size and transparency for weights with L1L2 regularization
plt.hist(weights_with_reg, bins=50, density=True, alpha=0.5, color='red', edgecolor='black', label='Weights with L1L2 Regularization')

# Plot the KDE of the weights with regularization using seaborn
sns.kdeplot(weights_with_reg, bw_adjust=0.5, color='red', label='KDE of Weights with L1L2 Regularization')

# Plot theoretical PDFs
plt.plot(x_vals, pdf_no_reg, 'b--', label=r'Theoretical PDF without Reg: $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{x^2}{2\sigma^2}}$')
plt.plot(x_vals, pdf_with_reg, 'r--', label=r'Theoretical PDF with L1L2 Reg: Combined Gaussian and Laplace')

# Add labels, title, and legend
plt.title('Weight Distributions with and without L1L2 Regularization')
plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend(loc='upper left')

# Save the plot to a file in the current working directory
plt.savefig('weight_distributions_L1L2_comparison.png')

# Show the plot
plt.show()
