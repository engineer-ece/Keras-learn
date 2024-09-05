import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# Create a sample 2D image (e.g., 6x6 grayscale image)
image = np.array([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 10, 11, 12],
                  [13, 14, 15, 16, 17, 18],
                  [19, 20, 21, 22, 23, 24],
                  [25, 26, 27, 28, 29, 30],
                  [31, 32, 33, 34, 35, 36]], dtype=float)

# Define a depthwise convolutional filter (2D)
depthwise_kernel = np.array([[1, 0],
                             [0, -1]], dtype=float)

# Depthwise convolution function
def depthwise_conv2d(image, kernel):
    kernel_size = kernel.shape[0]
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='constant')
    conv_result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            conv_result[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    return conv_result

# Initialize the figure and axis for the plot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Helper function to plot images
def plot_image(ax, data, title):
    im = ax.imshow(data, cmap='viridis', norm=Normalize(vmin=np.min(data), vmax=np.max(data)))
    ax.set_title(title)
    ax.axis('off')
    return im

# Initial plot setup
im1 = plot_image(axs[0], image, 'Original Image')
im2 = plot_image(axs[1], np.zeros_like(image), 'Depthwise Convolution Output')
im3 = plot_image(axs[2], np.zeros_like(image), 'Animated Depthwise Convolution')

# Animation function
def animate(frame):
    kernel_size = depthwise_kernel.shape[0]
    pad_size = kernel_size // 2
    
    # Calculate the current convolution result
    depthwise_conv_result = np.zeros_like(image)
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            depthwise_conv_result[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * depthwise_kernel)
    
    # Create the animated output by partially applying the convolution
    animated_output = np.zeros_like(image)
    for r in range(frame + 1):
        for c in range(frame + 1):
            if r < image.shape[0] and c < image.shape[1]:
                animated_output[r, c] = np.sum(padded_image[r:r+kernel_size, c:c+kernel_size] * depthwise_kernel)
    
    # Update images
    im2.set_data(depthwise_conv_result)
    im2.autoscale()
    im3.set_data(animated_output)
    im3.autoscale()
    
    return im2, im3

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=image.size, interval=500, blit=False)

# Display animation
plt.show()
