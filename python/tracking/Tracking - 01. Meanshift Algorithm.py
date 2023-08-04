# Import modules
import cv2

# Read videos
#video_capture = cv2.VideoCapture('/Users/hyejunghwang/datasets/0804-video_tracking.mp4')
video_capture = cv2.VideoCapture('/Users/hyejunghwang/datasets/0')

# Get a selected area by drag from users
return_, frame = video_capture.read()

bbox = cv2.selectROI('Selected Object',
                     frame,
                     fromCenter = False,
                     showCrosshair = True)
cv2.destroyAllWindows()
print('bbox information: ', bbox)


### Initialize tracking objects
# (x1, y1, x2, y2) == (bbox[0], bbox[1], bbox[2], bbox[3])
roi = frame[int(bbox[1]) : int(bbox[1] + bbox[3]),  # Vertical range of ROI
            int(bbox[0]) : int(bbox[0] + bbox[2])]  # Horizontal range of ROI

# Convert BGR to HSV
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

### Set a tracking of horizontal movement
# Set criteria to terminate mean shift iterations
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                 10,  # Stop after 10 iterations
                 1)   # Stop when the mean shift < 1

# Set a histogram of RoI
#  - Histogram provides generalized and robust representation of objects
#    as reduce a change of unrecognizable objects due to changes in lighting, angle, etc.
roi_hist = cv2.calcHist([roi_hsv],
                        [0],   # Hue channel  # Reason of using Hue channel
                                        #  - Robust to changes in lighting and shading
                                        #  - Hue is a stable characteristic as it's represented as color regardless its intensity
                        None,      # Mask
                        [180],   # Histogram size
                        [0,180])  # A range of colors: between 0 and 179
cv2.normalize(roi_hist,   # Start point
              roi_hist,   # End point
              0,    # Minimum value for normalization
              255,   # Maximum value for normalization
              cv2.NORM_MINMAX)  # Scaling method: Min-Max Normalization

# Track objects
while True:
    ret, frame = video_capture.read()

    ### Track objects in the current frame
    # Convert BGR to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Back project
    #  - Highlight regions in an images having similar colors or texture distribution to a given target region
    back_project = cv2.calcBackProject([frame_hsv], # Image source
                                       [0],        # '[0]': Hue channel
                                       roi_hist,            # Histogram of RoI
                                       [0, 120],     # A range of Hue channel: 0-179
                                       1)             # A scaling factor
    # Get results after mean shift
    ret, bbox = cv2.meanShift(back_project, bbox, term_criteria)
    print(bbox, bbox)

    # Display tracking
    x, y, w, h = bbox
    cv2.rectangle(frame,
                  (x, y),      # Top-left corner
                  (x+w, y+h),  # Bottom-right corner
                  (0, 255, 0), # Color
                  2)           # Thickness
    cv2.imshow('Tracking Test', frame)

    # Quit window when press 'q'
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# Run codes
video_capture.release()
cv2.destroyAllWindows()