Animating a `SeparableConv2D` operation involves visualizing the process of applying depthwise and pointwise convolutions on a 2D image. This can be split into two main steps:

1. **Depthwise Convolution**: Applies a separate convolutional filter to each channel independently.
2. **Pointwise Convolution**: Uses a 1x1 convolution to combine the outputs from the depthwise convolution.

Here's how you can create an animation to visualize these operations:

### SeparableConv2D Animation Code

```python
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

# Define a depthwise convolutional filter
depthwise_kernel = np.array([[1, 0],
                             [0, -1]], dtype=float)

# Define a pointwise convolutional filter
pointwise_kernel = np.array([[1, 1],
                              [1, 1]], dtype=float)

# Depthwise convolution function
def depthwise_conv2d(image, kernel):
    kernel_size = kernel.shape[0]
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='constant')
    conv_result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            conv_result[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    return conv_result

# Pointwise convolution function
def pointwise_conv2d(image, kernel):
    conv_result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            conv_result[i, j] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return conv_result

# Compute depthwise convolution
depthwise_output = depthwise_conv2d(image, depthwise_kernel)

# Compute pointwise convolution on depthwise output
pointwise_output = pointwise_conv2d(depthwise_output, pointwise_kernel)

# Initialize the figure and axis for the plot
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

# Helper function to plot images
def plot_image(ax, data, title):
    im = ax.imshow(data, cmap='viridis', norm=Normalize(vmin=np.min(data), vmax=np.max(data)))
    ax.set_title(title)
    ax.axis('off')
    return im

# Initial plot setup
im1 = plot_image(axs[0, 0], image, 'Original Image')
im2 = plot_image(axs[0, 1], depthwise_output, 'Depthwise Convolution Output')
im3 = plot_image(axs[0, 2], pointwise_output, 'Pointwise Convolution Output')

# Animation function
def animate(i):
    if i >= len(image):
        i = len(image) - 1
    
    # Animate depthwise convolution
    conv_depthwise = depthwise_conv2d(image[:i+1, :i+1], depthwise_kernel)
    im2.set_data(conv_depthwise)
    im2.autoscale()
    
    # Animate pointwise convolution
    conv_pointwise = pointwise_conv2d(conv_depthwise, pointwise_kernel)
    im3.set_data(conv_pointwise)
    im3.autoscale()
    
    return im2, im3

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=image.size, interval=500, blit=False)

# Display animation
plt.show()
```

### Explanation:

1. **Image Setup**:
   - The `image` is a 6x6 grayscale array to demonstrate the convolution.

2. **Kernel Definitions**:
   - `depthwise_kernel` is used for depthwise convolution.
   - `pointwise_kernel` is used for pointwise convolution.

3. **Convolution Functions**:
   - `depthwise_conv2d` applies the depthwise convolution.
   - `pointwise_conv2d` applies the pointwise convolution.

4. **Plotting**:
   - `plot_image` helps visualize the original image, depthwise convolution output, and pointwise convolution output.

5. **Animation**:
   - `animate(i)` updates the plots as the convolution process progresses.

### Running the Code:

1. **Ensure `matplotlib` is Installed**:
   ```bash
   pip install matplotlib
   ```

2. **Save the Script** to a file named `separable_conv2d_animation.py`.

3. **Run the Script**:
   ```bash
   python separable_conv2d_animation.py
   ```

### Notes:

- **Kernel Size**: Adjust the kernel sizes and image size to fit your needs. The given example uses simple 2x2 kernels and a 6x6 image for demonstration purposes.
- **Animation Frames**: The number of frames may need to be adjusted depending on your specific image and kernel sizes.

If the animation does not work or shows unexpected results, ensure that the environment where you are running the code supports `matplotlib` animations, and check for any specific errors that may arise.