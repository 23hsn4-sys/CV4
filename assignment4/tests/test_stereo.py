"""
Unit tests for stereo.py

Run with: pytest tests/test_stereo.py -v
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stereo import *


def create_synthetic_stereo_pair(height=50, width=100, disparity=10):
    """
    Create a synthetic stereo pair with known disparity for testing.
    
    The left image has a pattern that is shifted by 'disparity' pixels 
    to create the right image.
    """
    # Create a pattern
    np.random.seed(42)
    pattern = np.random.rand(height, width + disparity).astype(np.float32)
    
    # Left image: rightmost portion
    left = pattern[:, disparity:]
    
    # Right image: leftmost portion (shifted version)
    right = pattern[:, :width]
    
    return left, right, disparity


def test_compute_disparity_ssd():
    """
    Test SSD disparity computation on synthetic stereo pair.
    """
    # Create synthetic pair with known disparity
    left, right, true_disparity = create_synthetic_stereo_pair(
        height=50, width=100, disparity=15
    )
    
    # Compute disparity
    window_size = 5
    max_disparity = 30
    
    disparity_map = compute_disparity_ssd(left, right, window_size, max_disparity)
    
    # Check output shape
    assert disparity_map is not None, "disparity_map should not be None"
    assert disparity_map.shape == left.shape, f"Shape mismatch: {disparity_map.shape} vs {left.shape}"
    
    # Check that median disparity is close to true disparity
    # (excluding borders where matching may fail)
    half_win = window_size // 2
    center_region = disparity_map[half_win:-half_win, max_disparity:-half_win]
    median_disparity = np.median(center_region)
    
    assert abs(median_disparity - true_disparity) < 3, \
        f"Median disparity {median_disparity} too far from true {true_disparity}"


def test_compute_disparity_sad():
    """
    Test SAD disparity computation on synthetic stereo pair.
    """
    left, right, true_disparity = create_synthetic_stereo_pair(
        height=50, width=100, disparity=15
    )
    
    window_size = 5
    max_disparity = 30
    
    disparity_map = compute_disparity_sad(left, right, window_size, max_disparity)
    
    assert disparity_map is not None, "disparity_map should not be None"
    assert disparity_map.shape == left.shape
    
    half_win = window_size // 2
    center_region = disparity_map[half_win:-half_win, max_disparity:-half_win]
    median_disparity = np.median(center_region)
    
    assert abs(median_disparity - true_disparity) < 3, \
        f"Median disparity {median_disparity} too far from true {true_disparity}"


def test_compute_disparity_ncc():
    """
    Test NCC disparity computation on synthetic stereo pair.
    """
    left, right, true_disparity = create_synthetic_stereo_pair(
        height=50, width=100, disparity=15
    )
    
    window_size = 5
    max_disparity = 30
    
    disparity_map = compute_disparity_ncc(left, right, window_size, max_disparity)
    
    assert disparity_map is not None, "disparity_map should not be None"
    assert disparity_map.shape == left.shape
    
    half_win = window_size // 2
    center_region = disparity_map[half_win:-half_win, max_disparity:-half_win]
    median_disparity = np.median(center_region)
    
    assert abs(median_disparity - true_disparity) < 3, \
        f"Median disparity {median_disparity} too far from true {true_disparity}"


def test_disparity_to_depth():
    """
    Test depth conversion from disparity.
    """
    # Create a simple disparity map
    disparity_map = np.array([
        [10, 20, 30],
        [5, 10, 15],
        [2, 4, 8]
    ], dtype=np.float32)
    
    baseline = 0.1  # 10 cm
    focal_length = 500  # pixels
    
    depth_map = disparity_to_depth(disparity_map, baseline, focal_length)
    
    assert depth_map is not None, "depth_map should not be None"
    assert depth_map.shape == disparity_map.shape
    
    # Expected: depth = baseline * focal / disparity
    expected = baseline * focal_length / disparity_map
    
    # Handle potential inf values
    valid_mask = disparity_map > 0
    
    assert np.allclose(depth_map[valid_mask], expected[valid_mask], rtol=0.01), \
        "Depth values don't match expected formula"


def test_disparity_values_in_range():
    """
    Test that disparity values are within valid range.
    """
    left, right, _ = create_synthetic_stereo_pair(height=30, width=80, disparity=10)
    
    max_disparity = 25
    disparity_map = compute_disparity_ssd(left, right, 5, max_disparity)
    
    assert disparity_map.min() >= 0, "Disparity should be non-negative"
    assert disparity_map.max() <= max_disparity, \
        f"Disparity {disparity_map.max()} exceeds max {max_disparity}"
