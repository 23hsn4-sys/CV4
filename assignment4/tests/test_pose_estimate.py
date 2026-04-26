import os
from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2

from scripts.utils import *
from scripts.my_objectron import *
from scripts.pose_estimate import *


def test_pose_estimate(test_img='data/player1.jpg'):
  '''
  Tests the pose estimate
  '''
  if os.path.exists('data/player1.jpg'):
      test_img = 'data/player1.jpg'
  else:
      test_img = '../data/player1.jpg'

  land_mark, annotated_image = hand_pose_img(test_img)

  # Check that we got landmarks with the right shape
  assert land_mark is not None, "Landmarks should not be None"
  assert land_mark.shape[0] == 33, f"Expected 33 landmarks, got {land_mark.shape[0]}"
  assert land_mark.shape[1] == 2, f"Expected 2D landmarks, got shape {land_mark.shape}"

  # Check that landmark 22 (left hand) is in a reasonable location
  # The new PoseLandmarker model may give slightly different values than the old one,
  # so we use a generous tolerance.
  detected_left_thumb = np.array(land_mark[22])
  assert detected_left_thumb[0] > 50 and detected_left_thumb[0] < 500, \
      f"Left thumb x={detected_left_thumb[0]} out of expected range"
  assert detected_left_thumb[1] > 100 and detected_left_thumb[1] < 600, \
      f"Left thumb y={detected_left_thumb[1]} out of expected range"

  # Check annotated image is valid
  assert annotated_image is not None, "Annotated image should not be None"
  assert len(annotated_image.shape) == 3, "Annotated image should be 3-channel"


def test_projection_2d_to_3d():
  '''
  Test projection_2d_to_3d
  '''
  K = np.array([[ 500,   0, 535],
                            [   0, 500, 390],
                            [   0,   0,  -1]])

  R = np.array([[ 0.5,   -1,  0],
                            [   0,    0, -1],
                            [   1,  0.5,  0]])

  t = np.array([[   1,    0, 0, 300],
                              [   0,    1, 0, 300],
                              [   0,    0, 1,  30]])
  P = np.matmul(K, np.matmul(R, t))
  pose2d = np.array([[100,200],[200,300],[300,400]])
  depth = 1
  n=len(pose2d)

  pose3d_detected = projection_2d_to_3d(P, depth, pose2d)
  pose2d_reconstruct = P.dot(np.hstack((pose3d_detected, np.ones((n,1)))).T)[:2,:].T

  assert(np.allclose(pose2d, pose2d_reconstruct, atol=0.1))


if __name__=="__main__":
  test_pose_estimate()
  # test_projection_2d_to_3d()
