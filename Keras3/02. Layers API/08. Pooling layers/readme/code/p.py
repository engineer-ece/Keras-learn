import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.measure import block_reduce

# Original 6x6 image
image = np.array([[1, 2, 3, 0, 0, 1],
                  [4, 5, 6, 1, 1, 2],
                  [7, 8, 9, 2, 2, 3],
                  [0, 1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5, 6],
                  [2, 3, 4, 5, 6, 7]])

# Parameters for pooling
pool_size = (2, 2)
pool_height, pool_width = pool_size
num_rows, num_cols = image.shape
pooled_height = num_rows // pool_height
pooled_width = num_cols // pool_width

def apply_pooling(image, pool_type='max'):
    """Apply pooling operation and return the pooled image."""
    if pool_type == 'max':
        return block_reduce(image, pool_size, np.max)
    elif pool_type == 'avg':
        return block_reduce(image, pool_size, np.mean)
    else:
        raise ValueError("Unknown pool_type. Use 'max' or 'avg'.")

def highlight_region(image, row, col, pool_size, pool_type):
    """Highlight the pooling region."""
    h, w = pool_size
    region = image[row:row+h, col:col+w]
    if pool_type == 'max':
        return np.max(region)
    elif pool_type == 'avg':
        return np.mean(region)
    else:
        raise ValueError("Unknown pool_type. Use 'max' or 'avg'.")

def animate_pooling(i, ax1, ax2, ax3):
    """Update the plots for the animation."""
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # Original image with pooling regions highlighted
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Highlight the current pooling region
    row = (i // pooled_width) * pool_height
    col = (i % pooled_width) * pool_width
    
    if row + pool_height <= num_rows and col + pool_width <= num_cols:
        region = image[row:row+pool_height, col:col+pool_width]
        ax1.add_patch(plt.Rectangle((col, row), pool_width, pool_height, edgecolor='red', facecolor='none', linewidth=2))

    # Max pooling image
    pooled_image_max = apply_pooling(image, 'max')
    ax2.imshow(pooled_image_max, cmap='gray')
    ax2.set_title('MaxPooling')
    ax2.axis('off')
    
    # Average pooling image
    pooled_image_avg = apply_pooling(image, 'avg')
    ax3.imshow(pooled_image_avg, cmap='gray')
    ax3.set_title('AveragePooling')
    ax3.axis('off')
    
    return ax1, ax2, ax3

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

num_frames = pooled_height * pooled_width

ani = animation.FuncAnimation(
    fig, animate_pooling, frames=num_frames, 
    fargs=(ax1, ax2, ax3),
    interval=500, 
    blit=False
)

plt.show()
