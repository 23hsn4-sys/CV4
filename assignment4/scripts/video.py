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
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    K = calibrate(calib_path)
    
    annotated_frames = []
    vertices_world = get_world_vertices(size_x, size_y, size_z)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = img_rgb.shape[:2]
        
        K_scaled = K.copy()
        if calib_path and 'example2' in calib_path:
            K_scaled[0, 0] *= frame_w / 640.0
            K_scaled[1, 1] *= frame_h / 480.0
            K_scaled[0, 2] *= frame_w / 640.0
            K_scaled[1, 2] *= frame_h / 480.0
        
        try:
            bounding_boxes_2d, _ = detect_3d_box(frame)
        except:
            continue
        
        box_points_2d = np.array(bounding_boxes_2d)
        box_points_2d[:, 0] *= frame_w
        box_points_2d[:, 1] *= frame_h
        
        M_dlt, _ = estimate_camera_matrix(box_points_2d, vertices_world)
        
        K_inv = np.linalg.inv(K_scaled)
        R_approx = K_inv @ M_dlt[:, :3]
        U_r, _, Vt_r = np.linalg.svd(R_approx)
        R_init = U_r @ Vt_r
        if np.linalg.det(R_init) < 0:
            R_init = -R_init
        rvec_init, _ = cv2.Rodrigues(R_init)
        tvec_init = K_inv @ M_dlt[:, 3:]
        
        _, rvec, tvec = cv2.solvePnP(
            vertices_world.astype('float32'),
            box_points_2d.astype('float32'),
            K_scaled, None,
            rvec=rvec_init.astype('float32'),
            tvec=tvec_init.astype('float32'),
            useExtrinsicGuess=True
        )
        wRc_T, _ = cv2.Rodrigues(rvec)
        M = K_scaled @ np.hstack([wRc_T, tvec])
        
        landmark_2d, _ = hand_pose_img(path.replace('.mp4', '_frame.jpg'))
        if landmark_2d.shape[0] == 0:
            cv2.imwrite('/tmp/temp_frame.jpg', frame)
            landmark_2d, _ = hand_pose_img('/tmp/temp_frame.jpg')
        
        pose3d = projection_2d_to_3d(M, depth, landmark_2d)
        
        if pose3d.shape[0] > 22:
            hand_3d = pose3d[22]
            hand_3d = hand_3d.reshape(1, 3)
        else:
            hand_3d = np.zeros((1, 3))
        
        annotated = draw_box_intersection(img_rgb.copy(), hand_3d, vertices_world, bounding_boxes_2d)
        annotated_frames.append(annotated)
    
    cap.release()
    
    return annotated_frames if annotated_frames else [img_rgb]
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################
