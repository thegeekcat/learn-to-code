##############################################################
# Preparation
##############################################################

# Import modules
import cv2

# Load a video file
#file_path = './data/slow_traffic_small.mp4'
file_path = './data/traffic_1.mp4'
cap = cv2.VideoCapture(file_path)



##############################################################
# Initialization
##############################################################

# Set parameters for Shi-Tomasi Conner Detector
feature_params = dict(maxCorners = 100,   # Maximum number of coners to be detected
                      qualityLevel = 0.3, # Minimum quality level of coners to be detected
                      minDistance = 7,    # Minimum distance between coners -> '7': distance more than 7pixels
                      blockSize = 7)      # Size of neighborhood -> '7': 7x7 pixels
                                          

# Set parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize = (15, 15),                  # Size of window -> '15, 15': 15x15
                 maxLevel = 2,                        # M
                 criteria = (cv2.TERM_CRITERIA_EPS |  # Minimum value of moving vector between previous and currnet location
                             cv2.TERM_CRITERIA_COUNT, # Maximum number of iterations
                             10,                      # Maximum number of iterations
                             0.03))                   # Minimum value of displacement vector for termination



##############################################################
# Frameworks
##############################################################

# Read the first frame
ret, prev_frame = cap.read()

# Convert BGR to Gray
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Initialize a previous tracking position
prev_corners = cv2.goodFeaturesToTrack(prev_gray,         # Image source
                                       mask = None,
                                       **feature_params)  # Feature detection parameters set ealier

# Initialize a previous point
prev_points = prev_corners.squeeze()  # Flat the array to a 1D array


# Set colors for tracking results
color = (0, 255, 0)



##############################################################
# Track objects
##############################################################

while True:

    # Read the next frames
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Lucas-Kanade Optical Flow
    # status: '0': Tracked objects, '1': fail to track
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                      gray,
                                                      prev_points,
                                                      None,
                                                      **lk_params)
    
    # Show tracking results#
    for i, (prev_point, next_point) in enumerate(zip(prev_points, next_points)):
        x1, y1 = prev_point.astype(int)  # Coordinates of Previous point
        x2, y2 = next_point.astype(int)  # Coordinates of Next point

        # * zip: Combine multiple iterables into a single interable
        #      e.g. a = [1, 2, 3],  
        #           b = ['a', 'b', 'c']
        #           c = ['#', '$', '!']
        #           result = zip(a, b, c)
        #           for i in result:
        #               print(i)
        #           Output:
        #             (1, 'a', '#')
        #             (2, 'b', '$')
        #             (3, 'c', '!')

        # Draw a line
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        # Draw a circle
        cv2.circle(frame, 
                   (x2, y2), 
                   3,   # Radius of circle
                   color, 
                   -1)  # Negative thickness (Solid fill)

    # Visualization
    cv2.imshow('Feature-Based Tracking', frame)

    # Update variables for the next frames
    prev_gray = gray.copy()
    prev_points = next_points

    # Quit when 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()










##############################################################
# 
##############################################################