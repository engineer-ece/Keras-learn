import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Create a sample 3D volume (e.g., a stack of 3 2D images)
volume = np.array([[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],

                   [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],

                   [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]])

# Define a 3D convolutional filter
filter_kernel = np.array([[[1, 0],
                           [0, -1]],

                          [[1, 0],
                           [0, -1]],

                          [[1, 0],
                           [0, -1]]])

# Convolution parameters
filter_size = filter_kernel.shape
stride = 1
depth, height, width = volume.shape
filter_depth, filter_height, filter_width = filter_size
output_depth = (depth - filter_depth) // stride + 1
output_height = (height - filter_height) // stride + 1
output_width = (width - filter_width) // stride + 1

# Initialize the figure and axis for the plot
fig = plt.figure(figsize=(18, 10))

# Plot for the original volume
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.set_title('Original Volume')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Plot for the filter
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.set_title('Filter')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Plot for the output
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.set_title('Convolution Output')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

# Function to update plots
def update_plot(ax, data, title, vmin=None, vmax=None):
    ax.clear()
    x, y, z = np.indices(data.shape)
    sc = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=data.flatten(), cmap='viridis', s=100, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return sc

update_plot(ax1, volume, 'Original Volume', vmin=0, vmax=9)
filter_display = np.zeros_like(volume)
filter_img = update_plot(ax2, filter_display, 'Filter', vmin=-1, vmax=1)

def animate(i):
    z_start = i // (output_height * output_width)
    y_start = (i % (output_height * output_width)) // output_width
    x_start = i % output_width
    z_end = z_start + filter_depth
    y_end = y_start + filter_height
    x_end = x_start + filter_width
    
    # Compute convolution for the current position
    conv_result = np.sum(volume[z_start:z_end, y_start:y_end, x_start:x_end] * filter_kernel)
    
    # Update filter display
    filter_display.fill(0)
    filter_display[z_start:z_end, y_start:y_end, x_start:x_end] = filter_kernel
    sc_filter = update_plot(ax2, filter_display, 'Filter', vmin=-1, vmax=1)
    
    # Update output display
    feature_map = np.zeros((output_depth, output_height, output_width))
    feature_map[z_start // stride, y_start // stride, x_start // stride] = conv_result
    sc_output = update_plot(ax3, feature_map, 'Convolution Output', vmin=-1, vmax=1)
    
    return sc_filter, sc_output

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=output_depth * output_height * output_width, interval=500, blit=False)

# Display animation
plt.show()
