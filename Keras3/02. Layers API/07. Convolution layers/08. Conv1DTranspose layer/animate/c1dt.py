import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a sample 1D signal (e.g., a simple sequence)
input_signal = np.array([1, 2, 3, 4, 5], dtype=float)

# Define a Conv1DTranspose-like filter (1D)
# Here, we'll use a filter with stride 2 to demonstrate upsampling
filter_weights = np.array([0.5, 1, 0.5], dtype=float)

# Conv1DTranspose function
def conv1d_transpose(signal, kernel, stride):
    # Calculate the length of the output signal
    output_length = (len(signal) - 1) * stride + len(kernel)
    output_signal = np.zeros(output_length)
    
    # Apply the transposed convolution
    for i in range(len(signal)):
        start_index = i * stride
        output_signal[start_index:start_index+len(kernel)] += signal[i] * kernel
    
    return output_signal

# Initialize the figure and axis for the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the original signal
ax.plot(input_signal, label='Original Signal', color='b', linewidth=2)

# Animation function
def animate(frame):
    if frame >= len(input_signal) * 2:
        frame = len(input_signal) * 2 - 1
    
    # Compute Conv1DTranspose result
    upscaled_signal = conv1d_transpose(input_signal, filter_weights, stride=2)
    
    # Show upscaled signal up to current frame
    ax.clear()
    ax.plot(input_signal, label='Original Signal', color='b', linewidth=2)
    ax.plot(upscaled_signal[:frame+1], label='Conv1DTranspose Output', color='r', linestyle='--', linewidth=2)
    ax.fill_between(range(len(upscaled_signal[:frame+1])), upscaled_signal[:frame+1], color='r', alpha=0.3)
    ax.set_title('Conv1DTranspose Animation')
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=len(input_signal) * 2, interval=500, blit=False)

# Display animation
plt.show()
