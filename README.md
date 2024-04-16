# Real-Time Object Detection and Tracking 🐕‍🦺 

### This Python script performs real-time object detection and tracking using YOLOv5 for object detection. The script uses the OpenCV library for capturing video frames from a webcam and displaying the detection results.

## Dependencies
Python 3.x
PyTorch
OpenCV
numpy

## Installation
```
Install Python 3.x from python.org
Install PyTorch using pip install torch torchvision
Install OpenCV using pip install opencv-python
Install numpy using pip install numpy

```

## Usage
### Run the script.
```
python Filename.py
python3 Filename.py

```

### Run the script with command arguments.

```
python Filename.py --video path_to_video
```

To use parallel processing

```
python Filename.py --video path_to_video --parallel
```

To use Batch processing

```
python Filename.py --batch_size
```

The webcam feed will be displayed, and objects will be detected and tracked in real-time.

## Features

Real-time object detection using YOLOv5.

Display of bounding boxes and labels for detected objects.

Object Tracking: Tracks detected objects over consecutive frames using Kalman filters to predict and update their positions.

Parallel processing

## Configuration
window_width and window_height variables can be adjusted to change the size of the display window.
confidence_threshold variable can be adjusted to change the confidence threshold for object detection.

## References
```

YOLOv5: repository https://github.com/ultralytics/yolov5
KalmanFilter: https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
https://docs.python.org/3/library/multiprocessing.html

```