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

# Create a SIFT detector
sift = cv2.SIFT_create(contrastThreshold = 0.02)

# Set maximum number of keypoints
max_keypoints = 100



##############################################################
# Track objects
##############################################################

while True:

    # Read frames
    ret, frame = cap.read()
    if not ret:
        break

    # Transfer BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get keypoints and escriptors
    keypoints, descriptors = sift.detectAndCompute(        # detectAndCompute: Combine keypoints and descriptors
                                                    gray,
                                                    None)
    #print(keypoints, descriptors)
    #exit()
    
    
    # Limit number of keypoints
    if len(keypoints) > max_keypoints:
        keypoints = sorted(
                            keypoints, 
                            key = lambda x: -x.response   # 'lambda x: -x.response': sort keypoints by response in descending order
                          )[:max_keypoints]               # Keep top points

    # Draw keypoints
    frame = cv2.drawKeypoints(frame,
                              keypoints,
                              None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: Indicate keypoints to be drawn with size and orientataion
                                    # orientation -> Velocity of keypoints
    
    # Visualization
    cv2.imshow('Scale-Invariant Feature Transform', frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()







##############################################################
# 
##############################################################