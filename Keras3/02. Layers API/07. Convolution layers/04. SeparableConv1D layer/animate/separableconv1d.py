import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a sample 1D signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

# Define depthwise and pointwise kernels
depthwise_kernel = np.array([1, 0, -1], dtype=float)  # Example kernel
pointwise_kernel = np.array([1, 1, 1], dtype=float)   # Example kernel

# Depthwise convolution function
def depthwise_conv(signal, kernel):
    kernel_size = len(kernel)
    padded_signal = np.pad(signal, (kernel_size // 2, kernel_size // 2), mode='constant')
    conv_result = np.convolve(padded_signal, kernel, mode='valid')
    return conv_result

# Pointwise convolution function
def pointwise_conv(signal, kernel):
    conv_result = np.convolve(signal, kernel, mode='same')
    return conv_result

# Compute depthwise convolution
depthwise_output = depthwise_conv(signal, depthwise_kernel)

# Compute pointwise convolution on depthwise output
pointwise_output = pointwise_conv(depthwise_output, pointwise_kernel)

# Initialize the figure and axis for the plot
fig, axs = plt.subplots(3, 1, figsize=(12, 8))
line_width = 2
axs[0].plot(signal, label='Original Signal', color='b', linewidth=line_width)
axs[1].plot(depthwise_output, label='Depthwise Convolution Output', color='g', linewidth=line_width)
axs[2].plot(pointwise_output, label='Pointwise Convolution Output', color='r', linewidth=line_width)

for ax in axs:
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')

# Animation function
def animate(i):
    if i >= len(signal):
        i = len(signal) - 1

    # Depthwise Convolution
    depthwise_partial = depthwise_conv(signal[:i+1], depthwise_kernel)

    # Pointwise Convolution
    pointwise_partial = pointwise_conv(depthwise_partial, pointwise_kernel)

    # Update plots
    axs[0].plot(signal[:i+1], label='Original Signal', color='b', linewidth=line_width)
    axs[1].plot(depthwise_partial, label='Depthwise Convolution Output', color='g', linewidth=line_width)
    axs[2].plot(pointwise_partial, label='Pointwise Convolution Output', color='r', linewidth=line_width)
    
    axs[0].set_title('Original Signal')
    axs[1].set_title('Depthwise Convolution Output')
    axs[2].set_title('Pointwise Convolution Output')

    return axs

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=len(signal), interval=500, blit=False)

# Display animation
plt.show()
