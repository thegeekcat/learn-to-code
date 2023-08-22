# Import modules
import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load models
model = YOLO('./runs/detect/train/weights/best.pt')

# Open a video file
video_path = 'c:/datasets/0809-swimming_pool.mp4'
cap = cv2.VideoCapture(video_path)

# Store a tracking history
track_history = defaultdict(lambda : [])

while cap.isOpened():
    # Get variables
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, conf=0.5)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualization
        annotated_frame = results[0].plot()

        # Display track points
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]

            track.append((float(x), float(y)))

            if len(track) > 30:
                track.pop()

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(100, 100, 100), tickness=5)

        cv2.imshow('YOLOv8 Tracking Test', annotated_frame)

        if cv2.waitKey(30) & 0xff == ord('q'):
            break

    else:
        break
