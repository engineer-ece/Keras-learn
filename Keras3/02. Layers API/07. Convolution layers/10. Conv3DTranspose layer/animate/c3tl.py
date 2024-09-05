import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a sample 3D image (e.g., a 4x4x4 volume)
input_volume = np.array([[[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]],
                         
                         [[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]],
                         
                         [[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]],
                         
                         [[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]]], dtype=float)

# Define a Conv3DTranspose-like filter (3D)
filter_weights = np.array([[[1, 0],
                            [0, 1]],
                           
                           [[1, 0],
                            [0, 1]]], dtype=float)

# Conv3DTranspose function
def conv3d_transpose(volume, kernel, stride):
    kernel_size = kernel.shape[0]
    output_shape = ((volume.shape[0] - 1) * stride + kernel_size,
                    (volume.shape[1] - 1) * stride + kernel_size,
                    (volume.shape[2] - 1) * stride + kernel_size)
    output_volume = np.zeros(output_shape)
    
    # Apply the transposed convolution
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                start_i = i * stride
                start_j = j * stride
                start_k = k * stride
                output_volume[start_i:start_i+kernel_size, start_j:start_j+kernel_size, start_k:start_k+kernel_size] += volume[i, j, k] * kernel
    
    return output_volume

# Calculate the output volume size
stride = 2
conv3d_transpose_result = conv3d_transpose(input_volume, filter_weights, stride)

# Function to plot a 3D volume using scatter
def plot_3d_volume_scatter(volume, ax, title):
    x, y, z = np.nonzero(volume)
    ax.scatter(x, y, z, c=volume[x, y, z], cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Initialize the figure and axis for the 3D plot
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Plot the original and transposed convolution results
plot_3d_volume_scatter(input_volume, ax1, 'Original Volume')
plot_3d_volume_scatter(conv3d_transpose_result, ax2, 'Conv3DTranspose Output')

plt.show()
