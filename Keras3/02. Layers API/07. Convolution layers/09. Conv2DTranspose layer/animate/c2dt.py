import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a sample 2D image (e.g., a 4x4 grayscale image)
input_image = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]], dtype=float)

# Define a Conv2DTranspose-like filter (2D)
filter_weights = np.array([[1, 0],
                           [0, 1]], dtype=float)

# Conv2DTranspose function
def conv2d_transpose(image, kernel, stride):
    kernel_size = kernel.shape[0]
    output_shape = ((image.shape[0] - 1) * stride + kernel_size,
                    (image.shape[1] - 1) * stride + kernel_size)
    output_image = np.zeros(output_shape)
    
    # Apply the transposed convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            start_i = i * stride
            start_j = j * stride
            output_image[start_i:start_i+kernel_size, start_j:start_j+kernel_size] += image[i, j] * kernel
    
    return output_image

# Initialize the figure and axis for the plot
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Helper function to plot images
def plot_image(ax, data, title):
    im = ax.imshow(data, cmap='viridis', vmin=0, vmax=np.max(data))
    ax.set_title(title)
    ax.axis('off')
    return im

# Initial plot setup
im1 = plot_image(axs[0], input_image, 'Original Image')
output_shape = ((input_image.shape[0] - 1) * 2 + filter_weights.shape[0],
                (input_image.shape[1] - 1) * 2 + filter_weights.shape[1])
im2 = plot_image(axs[1], np.zeros(output_shape), 'Conv2DTranspose Output')

# Animation function
def animate(frame):
    stride = 2
    conv2d_transpose_result = conv2d_transpose(input_image, filter_weights, stride)
    
    # Create the animated output by partially applying the convolution
    rows = min(frame // output_shape[1] + 1, output_shape[0])
    cols = min(frame % output_shape[1] + 1, output_shape[1])
    
    animated_output = np.zeros_like(conv2d_transpose_result)
    animated_output[:rows, :cols] = conv2d_transpose_result[:rows, :cols]
    
    # Update images
    im2.set_data(animated_output)
    im2.autoscale()
    
    return im2,

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=output_shape[0] * output_shape[1], interval=200, blit=True)

# Display animation
plt.show()
