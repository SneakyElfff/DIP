import cv2
import numpy as np
import random

# Load the image
image_path = "/Users/nina/PycharmProjects/DIP/figures/1695128011387_processed.jpg"
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV range for blue
lower_blue = np.array([100, 100, 50])
upper_blue = np.array([140, 255, 255])

# Create a mask for blue regions
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Morphological operation to clean the mask
kernel = np.ones((29, 29), np.uint8)
cleaned_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

# Find contours to isolate the groups
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Prepare an output image
output_image = image.copy()

# Process each group independently
for i, contour in enumerate(contours):
    # Create a mask for the individual group
    group_mask = np.zeros_like(cleaned_mask)
    cv2.drawContours(group_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Enhance edges within the group mask (use Laplacian for subtle borders)
    laplacian = cv2.Laplacian(group_mask, cv2.CV_64F)
    enhanced_edges = cv2.convertScaleAbs(laplacian)

    # Combine enhanced edges with the original group mask
    combined_mask = cv2.bitwise_or(group_mask, enhanced_edges)

    # Apply distance transform
    dist_transform = cv2.distanceTransform(combined_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Find unknown regions (overlapping areas)
    unknown = cv2.subtract(group_mask, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Ensure background is not 0
    markers[unknown == 255] = 0

    # Apply watershed algorithm to the current group
    markers = cv2.watershed(output_image, markers)

    # Assign random colors to each separated object
    for marker_id in range(2, np.max(markers) + 1):  # Skip background and boundary
        random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        output_image[markers == marker_id] = random_color

# Save the final output
output_path = "/Users/nina/PycharmProjects/DIP/results/separated_objects_all_groups.jpeg"
cv2.imwrite(output_path, output_image)
print(f"All groups separated into individual objects. Saved to {output_path}")
