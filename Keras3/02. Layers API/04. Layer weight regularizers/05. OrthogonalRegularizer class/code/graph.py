import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import regularizers, initializers, models, layers

# Custom Regularizer class for Orthogonal Regularization
class CustomOrthogonalRegularizer(regularizers.Regularizer):
    def __init__(self, l=0.01):
        self.l = l

    def __call__(self, x):
        # TensorFlow operations for orthogonal regularization
        x = tf.convert_to_tensor(x)
        I = tf.eye(x.shape[1], dtype=x.dtype)
        orthogonal_loss = tf.reduce_sum(tf.square(tf.linalg.matmul(x, x, transpose_a=True) - I))
        return self.l * orthogonal_loss

    def get_config(self):
        return {'l': self.l}

# Function to create a model and extract weights
def create_model_with_orthogonal_regularization(l_value):
    model = models.Sequential([
        layers.Input(shape=(100,)),  # Define the input shape using Input layer
        layers.Dense(64, kernel_regularizer=CustomOrthogonalRegularizer(l=l_value),
                     kernel_initializer=initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Parameters for RandomNormal initializer
mean = 0.0
stddev = 0.05
seed = 42
l_value = 0.01  # Orthogonal regularization strength

# Create models and extract weights
model_custom_reg = create_model_with_orthogonal_regularization(l_value)

# Generate random input data
X = np.random.randn(1000, 100).astype(np.float32)
model_custom_reg.fit(X, np.random.randn(1000, 1).astype(np.float32), epochs=1, verbose=0)

# Extract weights
weights_with_reg = model_custom_reg.layers[1].get_weights()[0].flatten()

# Generate weights without regularization for comparison
initializer = initializers.RandomNormal(mean=mean, stddev=stddev, seed=seed)
weights_no_reg = initializer(shape=(100, 64)).numpy().flatten()

# Convert weights to NumPy arrays for easier plotting
weights_no_reg = np.array(weights_no_reg)
weights_with_reg = np.array(weights_with_reg)

# Theoretical PDF calculations
def normal_pdf(x, mean, stddev):
    return (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

# Define theoretical parameters
x_vals = np.linspace(-0.2, 0.2, 1000)

# Calculate theoretical PDFs
pdf_no_reg = normal_pdf(x_vals, mean, stddev)

# Orthogonal regularization does not change the fundamental distribution shape much, but it encourages orthogonality
# Here, we use normal distribution as an approximation
pdf_with_reg = normal_pdf(x_vals, mean, stddev)  

# Plot the histogram and KDE of the weights
plt.figure(figsize=(14, 7))

# Plot the histogram with custom bin size and transparency for weights without regularization
plt.hist(weights_no_reg, bins=50, density=True, alpha=0.5, color='blue', edgecolor='black', label='Weights without Regularization')

# Plot the KDE of the weights without regularization using seaborn
sns.kdeplot(weights_no_reg, bw_adjust=0.5, color='blue', label='KDE of Weights without Regularization')

# Plot the histogram with custom bin size and transparency for weights with Orthogonal regularization
plt.hist(weights_with_reg, bins=50, density=True, alpha=0.5, color='red', edgecolor='black', label='Weights with Orthogonal Regularization')

# Plot the KDE of the weights with regularization using seaborn
sns.kdeplot(weights_with_reg, bw_adjust=0.5, color='red', label='KDE of Weights with Orthogonal Regularization')

# Plot theoretical PDFs
plt.plot(x_vals, pdf_no_reg, 'b--', label=r'Theoretical PDF without Reg: $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{x^2}{2\sigma^2}}$')
plt.plot(x_vals, pdf_with_reg, 'r--', label=r'Theoretical PDF with Orthogonal Reg: Normal Distribution')

# Add labels, title, and legend
plt.title('Weight Distributions with and without Orthogonal Regularization')
plt.xlabel('Weight Value')
plt.ylabel('Density')
plt.grid(True)
plt.legend(loc='upper left')

# Save the plot to a file in the current working directory
plt.savefig('weight_distributions_orthogonal_comparison.png')

# Show the plot
plt.show()
