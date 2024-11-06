import cv2
import numpy as np

# Load the image
image_path = '/Users/nina/PycharmProjects/DIP/figures/1695128011387.jpg'  # Replace this with your image path
image = cv2.imread(image_path)
output = image.copy()

# Convert image to HSV color space for better isolation of blue tokens
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Define blue color range and create a mask
lower_blue = np.array([90, 50, 70])
upper_blue = np.array([240, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Apply morphological operations to remove noise and smooth the mask
kernel = np.ones((4, 4), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Use distance transform to separate overlapping objects
dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Identify unknown region (area between shapes)
unknown = cv2.subtract(mask, sure_fg)

# Marker labeling for watershed
num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  # Increment all labels by 1 so background is not 0
markers[unknown == 255] = 0  # Mark the unknown region with zero for watershed

# Apply the watershed algorithm to separate overlapping objects
markers = cv2.watershed(image, markers)
output[markers == -1] = [0, 0, 255]  # Outline boundaries with red color

# Draw outlines for each unique component group
for label in range(2, num_labels + 2):  # Skip the background label
    mask = np.uint8(markers == label)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 10)  # Green outline for each group

# Show the result
cv2.imshow("Separated Overlapping Shapes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
