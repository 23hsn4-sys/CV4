import os

# Suppress TF/MediaPipe C-level warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GLOG_minloglevel', '3')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scripts.utils import *

from scipy.ndimage.filters import maximum_filter

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def detect_peak(image, filter_size=3, order=0.5):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image,mask=~(image == local_max))

    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

def decode(hm, displacements):
    '''
    Decode the heatmap and displacement feilds from the encoder.
    Args:
        hm: heatmap of shape (1, 1, 40, 30)
        displacements: displacement fields of shape (1, 16, 40, 30)

    Returns:
        normalized vertices coordinates in 2D image
    '''
    hm = hm.reshape(hm.shape[2:])     # (40,30)

    peaks=hm.argmax()
    peakX = [peaks%30]
    peakY = [peaks//30]


    peaks=hm.argmax()
    peakX = [peaks%30]
    peakY = [peaks//30]

    scaleX = hm.shape[1]
    scaleY = hm.shape[0]
    objs = []
    for x,y in zip(peakX, peakY):
        conf = hm[y,x]
        points=[]
        for i in range(8):
            dx = displacements[0, i*2  , y, x]
            dy = displacements[0, i*2+1, y, x]
            points.append((x/scaleX+dx, y/scaleY+dy))
        objs.append(points)
    return objs


def draw_box(image, pts):
    '''
    Drawing bounding box in the image
    Args:
        image: image array
        pts: bounding box vertices

    Returns:

    '''
    scaleX = image.shape[1]
    scaleY = image.shape[0]

    lines = [(0,1), (1,3), (0,2), (3,2), (1,5), (0,4), (2,6), (3,7), (5,7), (6,7), (6,4), (4,5)]
    for line in lines:
        pt0 = pts[line[0]]
        pt1 = pts[line[1]]
        pt0 = (int(pt0[0]*scaleX), int(pt0[1]*scaleY))
        pt1 = (int(pt1[0]*scaleX), int(pt1[1]*scaleY))
        cv2.line(image, pt0, pt1, (255,0,0), thickness=10)

    for i in range(8):
        pt = pts[i]
        pt = (int(pt[0]*scaleX), int(pt[1]*scaleY))
        cv2.circle(image, pt, 8, (0,255,0), -1)
        cv2.putText(image, str(i), pt,  cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)


def detect_3d_box(img_path):

    '''
        Given an image, this function detects the 3D bounding boxes' 8 vertices of the chair in the image.
        We will only consider one chair in one single image.
        You should try to understand how the Objectron model works before trying to finish this function!
        Read the paper: https://arxiv.org/pdf/2003.03522.pdf

        This is an open-ended task — you are encouraged to consult the Objectron API docs:
        https://mediapipe.readthedocs.io/en/latest/solutions/objectron.html

        Args:
        -    img_path: the path of the RGB chair image

        Returns:
        -    boxes: list of 2D points (normalized), the 8 vertices of the 3D bounding box
        -    annotated_image: the original image with the overlapped bounding boxes

        Useful resources:
        -    tf.lite.Interpreter for loading the TFLite model
        -    The model file is a TFLite model that takes an image and outputs a heatmap + displacement fields
    '''

    # Search for model file in common locations
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_script_dir, 'object_detection_3d_chair.tflite'),
        'object_detection_3d_chair.tflite',
        'scripts/object_detection_3d_chair.tflite',
    ]
    model_path = None
    for _c in _candidates:
        if os.path.exists(_c):
            model_path = _c
            break
    if model_path is None:
        raise FileNotFoundError("Cannot find object_detection_3d_chair.tflite")

    boxes = None
    hm = None
    displacements = None

    inshapes = [[1, 3, 640, 480]]
    outshapes = [[1, 16, 40, 30], [1, 1, 40, 30]]


    if img_path == 'cam':
        cap = cv2.VideoCapture(0)

    while True:
        if img_path == 'cam':
            _, img_orig = cap.read()
        else:
            img_file = img_path
            img_orig = cv2.imread(img_file)


        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (inshapes[0][3], inshapes[0][2]))
        img = img.transpose((2,0,1))
        image = np.array(img, np.float32)/255.0

        ############################################################################
        # TODO: YOUR CODE HERE
        #
        # Load the TFLite model and run inference to obtain the heatmap and
        # displacement fields. Steps:
        #
        # 1. Load the TFLite model using tf.lite.Interpreter(model_path=model_path)
        #    and call allocate_tensors()
        #
        # 2. Get input_details and output_details from the interpreter
        #
        # 3. Prepare the input: create a float32 array matching input_shape,
        #    copy the 3 channels from `image` (which is CHW format) into it
        #    (the interpreter expects NHWC format), and set_tensor()
        #
        # 4. Call interpreter.invoke() to run inference
        #
        # 5. Extract the two output tensors using get_tensor():
        #    - One is the heatmap (shape ~[1, 40, 30, 1])
        #    - One is the displacement field (shape ~[1, 40, 30, 16])
        #
        # 6. Reshape the outputs so that:
        #    - hm has shape [1, 1, 40, 30]
        #    - displacements has shape [1, 16, 40, 30]
        #    (Hint: the raw outputs are in NHWC format, you need to rearrange)
        #
        # After this block, `hm` and `displacements` will be passed to decode().
        ############################################################################

        raise NotImplementedError("TODO: implement TFLite inference for Objectron")

        ############################################################################
        #                             END OF YOUR CODE
        ############################################################################
        # decode inference result
        boxes = decode(hm, displacements)

        # draw bbox
        for obj in boxes:
            draw_box(img_orig, obj)
        return boxes[0], cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
