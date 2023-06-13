##############################################################
# Preparation
##############################################################

# Import modules
import cv2

# Load images
img1 = cv2.imread('./data/cat9-3.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/cat9-2.jpg', cv2.IMREAD_GRAYSCALE)



##############################################################
# Initialization
##############################################################

# Create a ORB detector
orb = cv2.ORB_create()

# Calculate keypoints and descriptors
keypoints1, descriptor1 = orb.detectAndCompute(img1, None)
keypoints2, descriptor2 = orb.detectAndCompute(img2, None)



##############################################################
# Match keypoints
##############################################################

# Generate Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING,
                   crossCheck = True)

# Create matching objects
matches = bf.match(descriptor1, descriptor2)

# Sort matching results
matches = sorted(matches,
                 key=lambda x: x.distance)


##############################################################
# Visualization
##############################################################

# Draw a result
result = cv2.drawMatches(img1,
                         keypoints1,
                         img2,
                         keypoints2,
                         matches[:30],
                         None,
                         flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display
cv2.imshow('OBR and Brute-Force Algorithm', result)

# Calculate matching ratio
num_matches = len(matches)
num_good_matches = sum(1 for m in matches if m.distance < 50)
matching_percentage = (num_good_matches / num_matches) * 100

# Print result

print('Matching Ratio: %.2f%%' % matching_percentage)


cv2.waitKey(0)
cv2.destroyAllWindows()











##############################################################
# 
##############################################################