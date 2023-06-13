##############################################################
# Preparation
##############################################################

# Import modules
import numpy as np
import cv2

# Load images
img1 = cv2.imread('./data/cat9-3.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/cat9-2.jpg', cv2.IMREAD_GRAYSCALE)



##############################################################
# Initialization
##############################################################

# Create ORB detector
orb = cv2.ORB_create()



##############################################################
# Create FLANN matcher
##############################################################

# Calculate keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)


# Set parameters for indexing
index_params = dict(algorithm = 6,   # '6': LSH(Locality Sensitive Hasing) algorithm
                    table_number=25,
                    key_size=11,
                    multi_probe_level=1)

# Set a parameter for searching
search_params = dict(checks=20)  # number of searches when searching neighbors

# Create FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)




##############################################################
# Detect matches
##############################################################

# Find matches
matches = flann.knnMatch(descriptors1, 
                         descriptors2, 
                         k=2)


# Filter matching results
good_matches = []
for match in matches:
    if len(match) == 2:  # Check if the match contains two values
        m, n = match
        if m.distance < 0.77 * n.distance:
            good_matches.append(m)

# Draw matching results
result = cv2.drawMatches(img1,
                         keypoints1,
                         img2,
                         keypoints2,
                         good_matches[:30],
                         None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# Show results
match_ratio = len(good_matches) / len(matches) * 100
print(f'Matching Ratio: {match_ratio: .2f}%')


# Visualization
cv2.imshow('Fast Library for Approximate Nearest Neighbors Algorithm', result)
cv2.waitKey(0)
cv2.destroyAllWindows()










##############################################################
# 
##############################################################