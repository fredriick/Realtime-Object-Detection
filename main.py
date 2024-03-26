import cv2
import torch
from pathlib import Path
import time
from collections import defaultdict
import numpy as np
from datetime import datetime
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

# Define the window size
window_width = 900
window_height = 700

# Object class names
class_names = model.names

# Object counting dictionary
object_counts = defaultdict(int)

# Define colors for bounding boxes
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Confidence threshold
confidence_threshold = 0.5

# Recording variables
is_recording = False
video_writer = None

# Initialize FPS calculation
prev_time = 0

# Initialize Kalman filter for each object
kalman_filters = {}

# Define a function for real-time object detection
def detect_objects():
    global is_recording, video_writer
    global prev_time  # Declare prev_time as global
    
    cap = cv2.VideoCapture(0)  # Use the webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame
        frame = cv2.resize(frame, (window_width, window_height))
        
        # Perform inference
        results = model(frame, size=640)
        for _, pred in enumerate(results.pred):
            # Draw bounding boxes
            for det in pred:
                xmin, ymin, xmax, ymax, conf, cls = det.cpu().numpy()
                if conf > confidence_threshold:
                     # Initialize Kalman filter for new object
                    if cls not in kalman_filters:
                        kalman_filters[cls] = KalmanFilter(dim_x=4, dim_z=2)
                        kalman_filters[cls].x = np.array([xmin, ymin, 0, 0])
                        kalman_filters[cls].F = np.array([[1, 0, 1, 0],
                                                           [0, 1, 0, 1],
                                                           [0, 0, 1, 0],
                                                           [0, 0, 0, 1]])
                        kalman_filters[cls].H = np.array([[1, 0, 0, 0],
                                                           [0, 1, 0, 0]])
                        kalman_filters[cls].P *= 10
                        kalman_filters[cls].R = np.array([[10, 0],
                                                           [0, 10]])
                        kalman_filters[cls].Q = Q_discrete_white_noise(dim=4, dt=1, var=0.1)


                    # Predict object's next position with Kalman filter
                    kalman_filters[cls].predict()
                    
                    # Update object's position with detection
                    kalman_filters[cls].update(np.array([[xmin], [ymin]]))
                    
                    # Get predicted position
                    predicted_x, predicted_y = kalman_filters[cls].x[:2].astype(int)
                    
                    # Print predicted and updated positions
                    print(f'{class_names[int(cls)]}: Predicted Position ({predicted_x}, {predicted_y}), Updated Position ({xmin}, {ymin})')
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (predicted_x, predicted_y), (int(xmax), int(ymax)), (255, 0, 0), 2)
                    cv2.putText(frame, f'{class_names[int(cls)]}: {conf:.2f}', (predicted_x, predicted_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # cv2.imshow('Real-time Object Detection and Tracking', frame)

        color = colors[int(cls)]
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(frame, f'{model.names[int(cls)]}: {conf:.2f}', (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update object counts
        object_counts[class_names[int(cls)]] += 1

        # Display object counts
        count_text = ', '.join([f'{name}: {count}' for name, count in object_counts.items()])
        cv2.putText(frame, count_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Start or stop recording
        if is_recording:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(f'recorded_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            video_writer.write(frame)
        elif video_writer is not None:
            video_writer.release()
            video_writer = None

        # cv2.imshow('Real-time Object Detection', frame)    
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        
        # Display FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Real-time Object Detection', frame)
        
        key = cv2.waitKey(1)
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break
        elif key == ord('r'):  # Start/stop recording on 'r' key
            is_recording = not is_recording

    cap.release()
    cv2.destroyAllWindows()

# Run real-time object detection
detect_objects()
