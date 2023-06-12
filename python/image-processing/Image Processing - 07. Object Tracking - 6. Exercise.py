##############################################################
# Preparation
##############################################################

# Import modules
import cv2

# Load a video file
file_path = './data/traffic_1.mp4'
cap = cv2.VideoCapture(file_path)


##############################################################
# Initialization
##############################################################

# Create a SIFT detector
sift = cv2.SIFT_create(contrastThreshold=0.02)

# Set the maximum number of keypoints
#max_keypoints = 100

# Read the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Get keypoints as descriptors
prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_gray, None)



##############################################################
# Track objects
##############################################################

# Draw a bounding box: Select a target object  to track
bbox = cv2.selectROI('Selected Object',
                     prev_frame,
                     False,
                     False)

# Set an initial position
x, y, w, h = bbox
track_window = (x, y, w, h)
#print(x, y, w, h)

# Initialize the tracker
roi = prev_gray[y : y+h, x : x+w]
roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)

# Create a matcher
matcher = cv2.BFMatcher(cv2.NORM_L2)

# Create matching objects
matches = matcher.match(prev_descriptors, roi_descriptors)

# Sort matching results
matches = sorted(matches,
                 key=lambda x: x.distance)

# Get matching indexes
matching_indexes = [m.trainIdx for m in matches]



##############################################################
# Visualization
##############################################################

# Set colors
color = (0, 255, 0)



##############################################################
# Track objects
##############################################################



while True:

    # Read frames
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get tracking information
    roi_gray = gray[y: y+h, x: x+w]
    #print(x, y)
    roi_keypoints, roi_descriptors = sift.detectAndCompute(roi_gray, 
                                                           None)  # No mask

    # Track the same objects
    matches = matcher.match(prev_descriptors, roi_descriptors)

    # Sort matching results
    matches = sorted(matches, key=lambda x: x.distance)

    # Get matched indexes
    for match in matches:
        pt1 = prev_keypoints[match.queryIdx].pt    # pt1: Previous positions of descriptors
        pt2 = roi_keypoints[match.trainIdx].pt     # pt2: Current positions of descriptors
        x1, y1 = map(int, pt1)
        x2, y2 = map(int, pt2)
        cv2.circle(frame, (x + x2, y + y2), 3, color, -1)

    # Display
    cv2.imshow('Tracked Objects', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Update previous frames and tracking information
    prev_gray = gray.copy()
    prev_keypoints = roi_keypoints
    prev_descriptors = roi_descriptors


# Release resources
cap.release()
cv2.destroyAllWindows()





















##############################################################
# 
##############################################################