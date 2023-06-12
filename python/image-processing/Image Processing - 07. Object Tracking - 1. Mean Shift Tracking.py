# Install modules
#!pip install opencv-python==4.5.5.64


# Import modules
import cv2



##############################################################
# Set an initial rectangle for Mean Shift Tracking
##############################################################

# Set a variable for location
track_window = None  

# Set a variable for histogram of tracking object
roi_hist = None

# Set a criteria for Mean Shift  Tracking
term_crit = (cv2.TERM_CRITERIA_EPS |          # Minimum value of moving vector between previous and currnet location
             cv2.TERM_CRITERIA_COUNT,         # Maximum number of iterations
             10,                              # Maximum
             1)                               # Minimum



##############################################################
# Load a video file
##############################################################
# Load a video file
#file_path = './data/slow_traffic_small.mp4'
file_path = './data/traffic_1.mp4'
cap = cv2.VideoCapture(file_path)



##############################################################
# Preparation
##############################################################

# Select a target object from the first frame
ret, frame = cap.read()
x, y, w, h = cv2.selectROI('Selected ROI',   # Name of the Window
                           frame,         # image or video
                           False,         # 'True': Size of ROI is adjustable
                           False)         # 'True': A visiable rectangle around ROI
print('Coordinates of Selected Object: ', x, y, w, h)



##############################################################
# Calculate Initial Histogram
##############################################################

# Calculate initial histogram of tracking object
roi = frame[y : y+h, x : x+w]

# Save tracking object converted from BGR to HSV
# * HSV: Hue, Saturation, Value
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Calculate a histogram
roi_hist = cv2.calcHist([hsv_roi], # ROI Image
                        [0],       # Channel index -> [0]: Hue channel of HSV image
                        None,      # 'None': No mask applied
                        [180],     # Number of bins for histogram
                        [0, 100])  # Range of hue values [Min, Max]

# Normalize the histogram from 0 to 255
cv2.normalize(roi_hist,
              roi_hist,
              0,
              255,
              cv2.NORM_MINMAX)

# Set an initial window for tracking objects
track_window = (x, y, w, h)



##############################################################
# Track Objects
##############################################################

while True:
    # Read frames
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate Histogram Back Projection of Tracking object
    # Reason of using Back Projection: update object movement
    dst = cv2.calcBackProject([hsv],  
                              [0],      # Channel index -> [0]: Hue channel of HSV image
                              roi_hist, #
                              [0, 180], # Range of values for the Back Projection
                              1)        # Scaling factor
    
    # Estimate object location using Mean Shift Tracking
    ret, track_window = cv2.meanShift(dst,
                                      track_window,
                                      term_crit)
    
    # Show a ractangle of the tracking result
    x, y, w, h = track_window
    print(x, y, w, h)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Visualization
    cv2.imshow('Mean Shift Tracking', frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindws()


##############################################################
# Visualization
##############################################################
# Visualization
#cv2.imshow('ROI Test: ', roi)
#cv2.waitKey(0)




##############################################################
# 
##############################################################
