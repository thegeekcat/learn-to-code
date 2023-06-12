
##############################################################
# Preparation
##############################################################
# Import modules
import numpy as np
import cv2

# Load a video file
#video_path = './data/slow_traffic_small.mp4'
file_path = './data/traffic_1.mp4'
cap = cv2.VideoCapture(file_path)



##############################################################
# Initialize Kalman Filter
##############################################################

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)  # (Number of State variables, Number of Measurement variables)

# Measurement Matrix
#  - Measurement Matrix: Change State Vector to Measurement Vector
#  - A linear relationship between State Vector and measurement Vector
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]],
                                     np.float32)

# Transition Matrix
#  - Shows how State Vector changes over time 
#  - Predict the next State Vector based on the current State Vector (Markov)
kalman.transitionMatrix = np.array([[1, 0, 1, 0,],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]],
                                    np.float32)

# Covariance Matrix
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]],
                                   np.float32) * 0.05



##############################################################
# Set an Initial Object to track
##############################################################

# Select tracking object from the first frame
ret, frame = cap.read()
#print(ret, frame)

# Set a Bounding box
bbox = cv2.selectROI('Selected Object', frame, False, False)
print('Coordinates of Selected Object: ', bbox)

# Set an initial position
# * Velocity
#  - Object movement speed between frames
#  - Velocity = pixel/frame or m/sec        
kalman.statePre = np.array([[bbox[0]],  # X-axis of Top-Left conner
                            [bbox[1]],  # Y-axis of Top-Left conner
                            [0],        # Initial velocity along the X-axis
                            [0]],       # Initial velocity along the Y-axis
                            np.float32)



##############################################################
# Track objects
##############################################################
while True:

    # Read frames
    ret, frame = cap.read()
    if not ret:
        break

    # Estimate object position using Kalman Filter
    kalman.correct(
        # X-axis and Y-axis of the bounding box
        np.array([[np.float32(bbox[0] + bbox[2] / 2)],    # bbox[0]: X-axis, bbox[2]: width
                  [np.float32(bbox[1] + bbox[3] / 2)]]))  # bbox[1]: Y-axis, bbox[3]: height
    
    # Predict the next position
    kalman.predict()

    # Predicted position
    predicted_bbox = tuple(
                        map(int, 
                            kalman.statePost[:2, 0]))

    # Draw a rectangle for the position
    cv2.rectangle(
        frame,
        (predicted_bbox[0] - bbox[2] // 2, predicted_bbox[1] - bbox[3] // 2),
        (predicted_bbox[0] + bbox[2] // 2, predicted_bbox[1] + bbox[3] // 2),
        (0, 255, 0),
        2)
    
    # Visualization
    cv2.imshow('Kalman Filter Tracking', frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWIndows()








##############################################################
# 
##############################################################