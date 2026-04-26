"""
Binocular Stereo Matching Module

This module implements window-based stereo matching algorithms for rectified stereo pairs.
Students will implement SSD, SAD, and NCC matching functions to compute disparity maps.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_disparity_ssd(left_img, right_img, window_size, max_disparity):
    """
    Compute disparity map using Sum of Squared Differences (SSD).
    
    For each pixel in the left (reference) image, extract a window centered at that pixel.
    Then search along the corresponding scanline in the right image to find the best 
    matching window. The disparity is the horizontal offset that gives minimum SSD.
    
    Args:
        left_img: Left (reference) image, grayscale, shape (H, W)
        right_img: Right image, grayscale, shape (H, W)
        window_size: Size of the matching window (odd number, e.g., 5, 11, 21)
        max_disparity: Maximum disparity to search (positive integer)
    
    Returns:
        disparity_map: HxW disparity map where each pixel contains the disparity value
        
    Hints:
        - The disparity search should go from 0 to max_disparity
        - Handle border pixels appropriately (you can pad or skip them)
        - SSD formula: sum((left_window - right_window)^2)
        - For pixel (y, x) in left image, search (y, x-d) in right for d in [0, max_disparity]
    """
    
    disparity_map = None
    
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    
    raise NotImplementedError('`compute_disparity_ssd` function in '
                              'stereo.py needs to be implemented')
    
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    
    return disparity_map


def compute_disparity_sad(left_img, right_img, window_size, max_disparity):
    """
    Compute disparity map using Sum of Absolute Differences (SAD).
    
    Similar to SSD, but uses absolute differences instead of squared differences.
    SAD formula: sum(|left_window - right_window|)
    
    Args:
        left_img: Left (reference) image, grayscale, shape (H, W)
        right_img: Right image, grayscale, shape (H, W)
        window_size: Size of the matching window (odd number)
        max_disparity: Maximum disparity to search
    
    Returns:
        disparity_map: HxW disparity map
    """
    
    disparity_map = None
    
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    
    raise NotImplementedError('`compute_disparity_sad` function in '
                              'stereo.py needs to be implemented')
    
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    
    return disparity_map


def compute_disparity_ncc(left_img, right_img, window_size, max_disparity):
    """
    Compute disparity map using Normalized Cross-Correlation (NCC).
    
    NCC is more robust to brightness changes between the two images.
    The disparity is the offset that gives MAXIMUM NCC (not minimum like SSD/SAD).
    
    NCC formula: 
        NCC = sum((left_window - left_mean) * (right_window - right_mean)) / 
              (std_left * std_right * num_pixels)
    
    Args:
        left_img: Left (reference) image, grayscale, shape (H, W)
        right_img: Right image, grayscale, shape (H, W)  
        window_size: Size of the matching window (odd number)
        max_disparity: Maximum disparity to search
    
    Returns:
        disparity_map: HxW disparity map
        
    Hints:
        - NCC ranges from -1 to 1, with 1 being perfect match
        - Handle division by zero (when standard deviation is 0)
    """
    
    disparity_map = None
    
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    
    raise NotImplementedError('`compute_disparity_ncc` function in '
                              'stereo.py needs to be implemented')
    
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    
    return disparity_map


def disparity_to_depth(disparity_map, baseline, focal_length):
    """
    Convert disparity map to depth map.
    
    The relationship between disparity and depth for a rectified stereo pair is:
        depth = (baseline * focal_length) / disparity
    
    Args:
        disparity_map: HxW disparity map
        baseline: Distance between the two cameras (in same units as desired depth)
        focal_length: Focal length of the cameras (in pixels)
    
    Returns:
        depth_map: HxW depth map
        
    Hints:
        - Handle zero disparity (infinite depth) appropriately
        - You may want to set a maximum depth value for visualization
    """
    
    depth_map = None
    
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    
    raise NotImplementedError('`disparity_to_depth` function in '
                              'stereo.py needs to be implemented')
    
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
    
    return depth_map


# ============================================================================
# HELPER FUNCTIONS (PROVIDED - DO NOT MODIFY)
# ============================================================================

def load_stereo_pair(left_path, right_path):
    """
    Load a stereo pair and convert to grayscale.
    
    Args:
        left_path: Path to left image
        right_path: Path to right image
    
    Returns:
        left_gray: Left image in grayscale (float32, range 0-1)
        right_gray: Right image in grayscale (float32, range 0-1)
        left_color: Left image in color (for visualization)
    """
    left_color = cv2.imread(left_path)
    left_color = cv2.cvtColor(left_color, cv2.COLOR_BGR2RGB)
    right_color = cv2.imread(right_path)
    right_color = cv2.cvtColor(right_color, cv2.COLOR_BGR2RGB)
    
    left_gray = cv2.cvtColor(left_color, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    right_gray = cv2.cvtColor(right_color, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    return left_gray, right_gray, left_color


def visualize_disparity(disparity_map, title="Disparity Map"):
    """
    Visualize a disparity map using matplotlib.
    
    Args:
        disparity_map: HxW disparity map
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(disparity_map, cmap='jet')
    plt.colorbar(label='Disparity')
    plt.title(title)
    plt.axis('off')
    plt.show()


def compare_disparity_maps(disparity_maps, titles, figsize=(15, 5)):
    """
    Display multiple disparity maps side by side for comparison.
    
    Args:
        disparity_maps: List of disparity maps
        titles: List of titles for each map
        figsize: Figure size (width, height)
    """
    n = len(disparity_maps)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, (disp, title) in enumerate(zip(disparity_maps, titles)):
        im = axes[i].imshow(disp, cmap='jet')
        axes[i].set_title(title)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()


def visualize_depth_3d(depth_map, image, sample_rate=5, max_depth=None):
    """
    Visualize depth map as a 3D point cloud.
    
    Args:
        depth_map: HxW depth map
        image: HxW or HxWx3 color/grayscale image for coloring points
        sample_rate: Sample every nth pixel (for performance)
        max_depth: Maximum depth to display (clip larger values)
    """
    H, W = depth_map.shape
    
    # Create mesh grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Sample points
    u = u[::sample_rate, ::sample_rate].flatten()
    v = v[::sample_rate, ::sample_rate].flatten()
    z = depth_map[::sample_rate, ::sample_rate].flatten()
    
    # Get colors
    if len(image.shape) == 3:
        colors = image[::sample_rate, ::sample_rate].reshape(-1, 3)
    else:
        gray = image[::sample_rate, ::sample_rate].flatten()
        colors = np.stack([gray, gray, gray], axis=1)
    
    # Remove invalid points
    if max_depth is not None:
        valid = (z > 0) & (z < max_depth)
    else:
        valid = z > 0
    
    u, v, z = u[valid], v[valid], z[valid]
    colors = colors[valid]
    
    # Normalize colors to 0-1 if needed
    if colors.max() > 1:
        colors = colors / 255.0
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(u, z, -v, c=colors, s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Y')
    ax.set_title('3D Point Cloud from Depth Map')
    
    plt.show()


def measure_time(func, *args, num_runs=3, **kwargs):
    """
    Measure the average execution time of a function.
    
    Args:
        func: Function to time
        *args: Arguments to pass to function
        num_runs: Number of runs to average
        **kwargs: Keyword arguments to pass to function
    
    Returns:
        result: Result of the function
        avg_time: Average execution time in seconds
    """
    import time
    
    times = []
    result = None
    
    for _ in range(num_runs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    return result, avg_time
