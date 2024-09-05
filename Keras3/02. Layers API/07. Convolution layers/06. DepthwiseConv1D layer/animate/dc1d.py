import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a sample 1D signal
signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

# Define a depthwise convolutional filter (1D)
depthwise_kernel = np.array([1, 0, -1], dtype=float)

# Depthwise convolution function
def depthwise_conv1d(signal, kernel):
    kernel_size = len(kernel)
    padded_signal = np.pad(signal, (kernel_size // 2, kernel_size // 2), mode='constant')
    conv_result = np.zeros_like(signal)
    for i in range(len(signal)):
        conv_result[i] = np.sum(padded_signal[i:i+kernel_size] * kernel)
    return conv_result

# Initialize the figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the original signal
ax.plot(signal, label='Original Signal', color='b', linewidth=2)

# Animation function
def animate(i):
    if i >= len(signal):
        i = len(signal) - 1
    
    # Apply depthwise convolution filter
    conv_result = depthwise_conv1d(signal[:i+1], depthwise_kernel)
    
    # Clear the axis and plot the results
    ax.clear()
    ax.plot(signal, label='Original Signal', color='b', linewidth=2)
    ax.plot(conv_result, label='Depthwise Convolution Output', color='r', linestyle='--', linewidth=2)
    ax.fill_between(range(len(conv_result)), conv_result, color='r', alpha=0.3)
    ax.set_title('Depthwise Conv1D Animation')
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=len(signal), interval=500, blit=False)

# Display animation
plt.show()
