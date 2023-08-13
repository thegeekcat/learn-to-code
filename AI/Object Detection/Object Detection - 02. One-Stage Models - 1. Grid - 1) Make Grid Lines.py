# Import modules
import cv2
import numpy as np

# Load images
image = cv2.imread('./data/llama-02.png')
#print(image)

# Set size of gird cells
grid_size = (20, 20)

# Define a function to create grids
def create_grid(image, grid_size=grid_size):
    height, width = image.shape[:2]
    print(height, width)
    grid_width, grid_height = grid_size
    print(grid_width, grid_height)

    # Create grids
    grid_image = np.copy(image)
    for x in range(0, width, grid_width):
        cv2.line(grid_image, (x, 0), (x, height), (255, 255, 255), 1)
    for y in range(0, height, grid_height):
        cv2.line(grid_image, (0, y), (width, y), (255, 255, 255), 1)

    return grid_image

# Run codes
grid_image = create_grid(image, grid_size)

cv2.imshow('Original Image', image)
cv2.imshow('Grid Image', grid_image)
cv2.waitKey(0)
cv2.imshow()