import cv2
import torch
from pathlib import Path
import time

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the device (GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

# Define the window size
window_width = 900
window_height = 700

# Initialize FPS calculation
prev_time = 0

# Define a function for real-time object detection
def detect_objects():
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
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                cv2.putText(frame, f'{model.names[int(cls)]}: {conf:.2f}', (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        
        # Display FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Real-time Object Detection', frame)
        
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time object detection
detect_objects()
