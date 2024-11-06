import cv2
import numpy as np
import os

# Define paths
input_folder = '/Users/nina/PycharmProjects/DIP/figures'
output_folder = '/Users/nina/PycharmProjects/DIP/results'

# Define HSV range to isolate blue color - adjusted range
lower_blue = np.array([90, 30, 50])
upper_blue = np.array([240, 255, 255])

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each file in the input folder
for filename in os.listdir(input_folder):
    # Only process files with typical image extensions
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Construct full file paths
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.', '_processed.'))

        # Load the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Skipping file {filename}: cannot read.")
            continue

        # Convert the image to HSV color space
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask for blue color
        mask_blue = cv2.inRange(image_hsv, lower_blue, upper_blue)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

        # Use distance transform with adjusted threshold
        dist_transform = cv2.distanceTransform(mask_cleaned, cv2.DIST_L2, 5)
        _, dist_thresh = cv2.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)
        dist_thresh = np.uint8(dist_thresh)

        # Find sure background and sure foreground
        sure_bg = cv2.dilate(mask_cleaned, kernel, iterations=3)
        sure_fg = dist_thresh

        # Subtract sure foreground from sure background to get unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label markers for watershed
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply watershed algorithm
        markers = cv2.watershed(image, markers)
        output_image = image.copy()

        # Find and draw contours
        for marker in np.unique(markers):
            if marker <= 1:  # Ignore background marker
                continue

            # Create a mask for each separated object
            mask = np.zeros(markers.shape, dtype="uint8")
            mask[markers == marker] = 255

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:  # Adjust if needed based on token size
                    cv2.drawContours(output_image, [cnt], -1, (0, 0, 255), 10)

        # Save the result
        cv2.imwrite(output_path, output_image)
        print(f"Processed and saved: {output_path}")
