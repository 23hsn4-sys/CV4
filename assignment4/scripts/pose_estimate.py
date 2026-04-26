import os

# Suppress C-level warnings from mediapipe/TF
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GLOG_minloglevel', '3')

import mediapipe as mp
import numpy as np
from PIL import Image
import cv2
from scripts.utils import *


# Path to the pose landmarker model (downloaded during setup)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, 'pose_landmarker_heavy.task'),
    'scripts/pose_landmarker_heavy.task',
    'pose_landmarker_heavy.task',
]


def _get_model_path():
    for c in _MODEL_CANDIDATES:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Cannot find pose_landmarker_heavy.task. "
        "Download it with: wget -O scripts/pose_landmarker_heavy.task "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    )


def hand_pose_img(test_img):
    """
        Given an image, it calculates the pose of human in the image.
        To make things easier, we only consider one person in a single image.
        Uses the MediaPipe Tasks API (PoseLandmarker).

        Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

        Args:
        -    test_img: path to rgb image

        Returns:
        -    landmark: numpy array of size (n, 2), the landmarks detected by mediapipe,
             where n is the number of landmarks, 2 represents x and y coordinates
             in pixel space (not normalized 0-1).
        -    annotated_image: the original image overlapped with the detected landmarks

        Useful classes: mp.tasks.vision.PoseLandmarker, mp.tasks.vision.PoseLandmarkerOptions
    """

    landmark = None
    annotated_image = None

    # Read the image
    image = cv2.imread(test_img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows = image.shape[0]
    cols = image.shape[1]

    ############################################################################
    # TODO: YOUR CODE HERE
    #
    # Use the MediaPipe Tasks API to detect pose landmarks, then convert
    # the normalized landmark coordinates to pixel coordinates.
    #
    # Part A — Detection:
    #   1. Create PoseLandmarkerOptions with the model path from _get_model_path()
    #      Hint: use mp.tasks.BaseOptions, mp.tasks.vision.PoseLandmarkerOptions,
    #            and mp.tasks.vision.RunningMode.IMAGE
    #   2. Create a PoseLandmarker using create_from_options()
    #   3. Create an mp.Image from the file using mp.Image.create_from_file()
    #   4. Call landmarker.detect() to get the result
    #
    # Part B — Landmark conversion:
    #   The result object has result.pose_landmarks which is a list of detected
    #   poses. Each pose is a list of landmarks with .x, .y attributes
    #   (normalized to [0, 1] range).
    #
    #   Convert the first detected pose's landmarks to pixel coordinates:
    #   - Create a numpy array `landmark1` of shape (n, 2)
    #   - For each landmark, multiply .x by `cols` and .y by `rows`
    #   - If no pose is detected, set landmark1 = np.zeros((0, 2))
    #
    # Store the detection result in a variable called 'result'.
    # Store the pixel-coordinate landmarks in a variable called 'landmark1'.
    ############################################################################

    model_path = _get_model_path()
    
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
    
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(test_img)
        result = landmarker.detect(mp_image)
        
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks_normalized = result.pose_landmarks[0]
            landmark1 = np.array([[lm.x * cols, lm.y * rows] for lm in landmarks_normalized])
        else:
            landmark1 = np.zeros((0, 2))
    
    landmark = landmark1

    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    # Draw landmarks on the image (provided — do not modify)
    annotated_image = image.copy()
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        detected_landmarks = result.pose_landmarks[0]
        num_landmarks = len(detected_landmarks)
        for i, lm in enumerate(detected_landmarks):
            cx, cy = int(lm.x * cols), int(lm.y * rows)
            cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)

        # Draw connections between landmarks
        connections = [
            (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
            (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
            (12,14),(14,16),(16,18),(16,20),(16,22),
            (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
            (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
        ]
        for start_idx, end_idx in connections:
            if start_idx < num_landmarks and end_idx < num_landmarks:
                start_lm = detected_landmarks[start_idx]
                end_lm = detected_landmarks[end_idx]
                start_pt = (int(start_lm.x * cols), int(start_lm.y * rows))
                end_pt = (int(end_lm.x * cols), int(end_lm.y * rows))
                cv2.line(annotated_image, start_pt, end_pt, (0, 255, 0), 2)

    return landmark1, annotated_image
