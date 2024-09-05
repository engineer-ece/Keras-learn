import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a sample 2D image
image = np.array([[1, 2, 3, 0],
                  [4, 5, 6, 1],
                  [7, 8, 9, 2],
                  [0, 1, 2, 3]])

# Define a 2D convolutional filter
filter_kernel = np.array([[1, 0],
                          [0, -1]])

# Convolution parameters
filter_size = filter_kernel.shape
stride = 1
image_height, image_width = image.shape
filter_height, filter_width = filter_size
output_height = (image_height - filter_height) // stride + 1
output_width = (image_width - filter_width) // stride + 1

# Initialize the figure and axis for the plot
fig, ax = plt.subplots(1, 3, figsize=(14, 6))

# Plot for the original image
ax[0].imshow(image, cmap='gray', vmin=0, vmax=9)
ax[0].set_title('Original Image')
ax[0].axis('off')

# Plot for the filter
filter_display = np.zeros((image_height, image_width))
filter_img = ax[1].imshow(filter_display, cmap='gray', vmin=-1, vmax=1)
ax[1].set_title('Filter')
ax[1].axis('off')

# Plot for the output
output_display = np.zeros((output_height, output_width))
output_img = ax[2].imshow(output_display, cmap='gray', vmin=-2, vmax=2)
ax[2].set_title('Convolution Output')
ax[2].axis('off')

def init():
    output_img.set_data(np.zeros((output_height, output_width)))
    filter_img.set_data(np.zeros((image_height, image_width)))
    return filter_img, output_img

def animate(i):
    # Calculate the current position of the filter
    start_x = (i % output_width) * stride
    start_y = (i // output_width) * stride
    end_x = start_x + filter_width
    end_y = start_y + filter_height
    
    # Compute convolution for the current position
    conv_result = np.sum(image[start_y:end_y, start_x:end_x] * filter_kernel)
    
    # Create filter display
    filter_display = np.zeros((image_height, image_width))
    filter_display[start_y:end_y, start_x:end_x] = filter_kernel
    filter_img.set_data(filter_display)
    
    # Create output feature map
    feature_map = np.zeros((output_height, output_width))
    feature_map[start_y // stride, start_x // stride] = conv_result
    output_img.set_data(feature_map)
    
    return filter_img, output_img

# Create animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=output_height * output_width, interval=500, blit=True)

# Display animation
plt.show()
