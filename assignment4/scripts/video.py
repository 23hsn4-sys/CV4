import os

# Suppress C-level TF/MediaPipe warnings (must be set before importing TF/mediapipe)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2
from scripts.my_objectron import *
from scripts.pnp import estimate_camera_matrix, camera_center
from scripts.calibration import calibrate
from scripts.intersection import *
from scripts.pose_estimate import *
from scripts.utils import *


def process_video(path, num_frames=20, depth=1.91,
                   calib_path='data/cali2/example2/',
                   size_x=0.4, size_y=0.4, size_z=1.0):
    """
    Process a video by sampling frames and running the full pipeline
    (objectron detection, DLT + refinement, pose estimation, intersection check).

    Args:
        path: path to the video file (.mp4)
        num_frames: number of evenly-spaced frames to sample (default 20)
        depth: estimated depth (distance) of the person from the camera in meters.
               You need to adjust this for your own video/setup.
        calib_path: path to calibration images (needed for DLT refinement)
        size_x, size_y, size_z: chair dimensions in meters

    Returns:
        annotated_frames: list of annotated RGB numpy arrays (one per sampled frame)

    Approach:
    1. Open the video with cv2.VideoCapture
    2. Sample `num_frames` evenly-spaced frames using np.linspace
    3. Calibrate camera (K) and scale if video resolution differs from calibration
    4. For each sampled frame:
       a. Run detect_3d_box to get 2D bounding box points
       b. Scale the normalized box points to pixel coordinates
       c. Run estimate_camera_matrix (DLT) to get initial M
       d. Refine M using cv2.solvePnP with DLT as initial guess
       e. Run hand_pose_img to get 2D pose landmarks
       f. Back-project 2D landmarks to 3D using projection_2d_to_3d
       g. Use draw_box_intersection to annotate the frame
    5. Collect all annotated frames in a list and return it
    """

    inside = None
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
