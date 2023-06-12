##############################################################
# Preparation
##############################################################

# Import modules
import cv2
import numpy as np

# Load a video file
#file_path = './data/slow_traffic_small.mp4'
file_path = './data/traffic_1.mp4'
cap = cv2.VideoCapture(file_path)



##############################################################
# Initialization
##############################################################

# Create an ORB object
orb = cv2.ORB_create()

# Set minimum size of keypoints
min_keypoint_size = 10

# Set a distance to remove duplicated keypoints
duplicate_threshold = 10


##############################################################
# Track Object
##############################################################

while True:

    # Read frames
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints
    keypoints = orb.detect(gray, None)
    
    # Remain keypoints only above threshold
    keypoints = [kp for kp in keypoints if kp.size > min_keypoint_size]
                # * List Comprehension
                #   : [EXPRESSION for ITEMS in INTERABLE if CONDITION]
                #   - `for statement` + `if statement`
                #     a. `for statement` -> `for kp in keypoints:
                #     b. `if statement` -> `if kp.size > min_keypoint_size:`
                #   - 

    
    # Create a mask to remove duplicated keypoints
    mask = np.ones(len(keypoints), dtype = bool)  # 'True' in default
    
    # Remove duplicated keypoints
    for i, kp1 in enumerate(keypoints):
        if mask[i]:
            for j, kp2 in enumerate(keypoints[i+1 : ]):   # Avoid comparison with itself and duplicated keypoints
                if (
                    #print(kp1.pt, kp2.pt)  # `pt`: Axes of point coordinates
                    mask[i + j + 1] 
                    and np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt)) < duplicate_threshold):
                        # => np.linalg.nor(np.array(kp1.pt) - np.array(kp2.pt)): Euclidean distance between kp1 and kp2

                        mask[i + j + 1] = False # Duplicated keypoints -> mask='False' 

    # Create a new list of keypoints without duplicated keypoints(mask='False')
    keypoints = [ kp for i, kp in enumerate(keypoints) if mask[i]]

    # Draw keypoints
    frame = cv2.drawKeypoints(frame, 
                              keypoints, 
                              None,    # Output image -> 'None': keypoints will be drawn on the image itself
                              color=(0, 255, 0), 
                              flags=0)

    # Display
    cv2.imshow('Oriented Fast and Rotated BRIEF', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()








##############################################################
# 
##############################################################