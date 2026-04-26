# Assignment 4: 3D Pose Estimation, Objectron, and Stereo Matching

## 1) Environment Setup (Conda)

Create and activate a fresh Conda environment:

```bash
conda create -n cv python=3.10 -y
conda activate cv
```

Install dependencies:

```bash
pip install numpy scipy matplotlib pillow opencv-python tqdm
pip install torch torchvision
pip install tensorflow mediapipe
pip install jupyterlab notebook ipykernel
```

Optional (recommended) — register this environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name cv --display-name "Python (cv)"
```

Notes:
- Windows users: MediaPipe may not have good support. Consider using macOS/Linux or Google Colab.
- Mac users: if MediaPipe installation fails, follow the [official instructions](https://ai.google.dev/edge/mediapipe/solutions/guide) or use Google Colab.

## 2) Model Downloads

Download the MediaPipe Pose Landmarker model (required for Part 3):

```bash
wget -O scripts/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

The Objectron TFLite model (`scripts/object_detection_3d_chair.tflite`) is already included.

## 3) Data

All data is provided in the `data/` folder:
- `data/10.jpg`, `data/chair.jpg`, `data/player1.jpg` — sample images for testing the pipeline
- `data/cali2/example2/` — example calibration images
- `data/stereo_pairs/` — rectified stereo image pairs (Tsukuba)
- `data/Checkerboard-A4-35mm-7x4.pdf` — checkerboard for camera calibration
- `data/cali2/video.mp4` — sample video (for reference only; you must record your own video for Part 7)

**Note:** For Parts 6 and 7, you must use your own captured images/videos. You will receive zero credit if you only demonstrate results on the provided samples.

## 4) Launch

```bash
jupyter lab
```

Then open the main notebook: `assignment4.ipynb`

## 5) Where to Implement

All code TODOs are in the `scripts/` folder. Here is the complete list:

| TODO | Points | File | Function |
|------|--------|------|----------|
| TODO 1 | 10 pts | `scripts/my_objectron.py` | `detect_3d_box()` |
| TODO 2a | 10 pts | `scripts/utils.py` | `get_world_vertices()` |
| TODO 2b | 10 pts | `scripts/pnp.py` | `camera_center()`, `project()`, `reprojection_error()`, `estimate_camera_matrix()` |
| TODO 3 | 10 pts | `scripts/pose_estimate.py` | `hand_pose_img()` |
| TODO 4 | 15 pts | `scripts/utils.py` | `projection_2d_to_3d()` |
| TODO 5 | 15 pts | `scripts/intersection.py` | `check_hand_inside_bounding_box()` |
| TODO 7 | 10 pts | `scripts/video.py` | `process_video()` |
| TODO 8a | 15 pts | `scripts/stereo.py` | `compute_disparity_ssd()` |
| TODO 8b | 10 pts | `scripts/stereo.py` | `compute_disparity_sad()` |
| TODO 8c | 15 pts | `scripts/stereo.py` | `compute_disparity_ncc()` |
| TODO 8d | 7 pts | `scripts/stereo.py` | `disparity_to_depth()` |

### Grading Summary

| Part | Points |
|------|--------|
| Part 1: 3D Bounding Box Detection (Objectron) | 15 |
| Part 2: Camera Calibration & Projection Matrix (DLT) | 20 |
| Part 3: Human Pose Estimation | 15 |
| Part 4: 2D to 3D Projection | 15 |
| Part 5: Intersection Detection | 15 |
| Part 6: Own Image | 10 |
| Part 7: Video Intersection Detection | 10 |
| Part 8: Binocular Stereo + Depth (SSD + SAD + NCC + Depth) | 52 |
| **Total** | **152** |

## 6) Testing

Run unit tests from the assignment root:

```bash
pytest tests/ -v
```

Tests are also embedded in the notebook for checking progress as you work.

## 7) Submission

Create your submission zip from the assignment root:

```bash
bash submit.sh
```

This generates `assignment4_submission.zip` and excludes `data/` folders.

Submit:
- The generated zip file
- A PDF report with answers to all written/inline notebook questions
- At the top of your PDF, include AI usage disclosure (if any): tool used + how you used it

## 8) Policy on the Use of AI Assistants

You are welcome to use AI tools in this course in the same way you might use office hours or ask clarifying questions from an instructor. Appropriate AI uses include:

- Clarifying lecture concepts
- Understanding problem statements
- Getting high-level guidance or tips on how to get started

However, just as you would not ask another person to complete your assignment for you, **you should not ask an AI to write code, do your work, or provide full solutions**.

When in doubt, imagine the AI as a human helper and apply the same academic integrity standards.

If you use any AI assistance while working on a problem set, please include a short disclosure at the top of your submission describing:

- Which AI tool you used
- How you used it

A few sentences is sufficient.
