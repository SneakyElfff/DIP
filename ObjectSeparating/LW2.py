import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define paths
input_folder = '/Users/nina/PycharmProjects/DIP/figures'
output_folder = '/Users/nina/PycharmProjects/DIP/results'

# Define HSV range to isolate blue color
lower_blue = np.array([90, 30, 50])
upper_blue = np.array([240, 255, 255])

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Area threshold for filtering out small, inner contours
min_contour_area = 1000  # Adjust this value as needed based on your images


# Method to check if a contour matches a standard shape
def is_standard_shape(contour):
    # Approximate the contour to simplify it
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the contour is a circle, square, or hexagon
    area = cv2.contourArea(contour)
    if area < min_contour_area:
        return False

    # Check if the contour is circular
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / perimeter ** 2)
    if 0.7 <= circularity <= 1.2:  # Adjust the range for circularity
        return True

    # Check for square or hexagon based on vertices
    if len(approx) == 4:  # Square
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.4:
            return True
    elif len(approx) == 6:  # Hexagon
        return True

    return False


# Method to process an image using combined contours or method 2
def process_image(file_path, output_folder):
    image = cv2.imread(file_path)
    if image is None:
        print(f"Skipping file {file_path}: cannot read.")
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply morphological operations
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Method 2 (simple contour extraction)
    contours_2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours_2 = [cnt for cnt in contours_2 if is_standard_shape(cnt)]

    # If there are no overlapping objects, use method 2's contours
    if len(filtered_contours_2) == len(contours_2):
        print("No overlap detected, using method 2 contours.")
        result_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.drawContours(result_image, filtered_contours_2, -1, (0, 255, 255), 10)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        cv2.imwrite(os.path.join(output_folder, f'{file_name}_method2.jpg'), result_image)
    else:
        print("Overlap detected, using combined approach from method 1 and method 2.")

        # Method 1: Using Distance Transform with 0.5 threshold
        dist_transform_1 = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, sure_fg_1 = cv2.threshold(dist_transform_1, 0.5 * dist_transform_1.max(), 255, 0)
        sure_fg_1 = np.uint8(sure_fg_1)
        unknown_1 = cv2.subtract(mask, sure_fg_1)
        num_labels_1, markers_1 = cv2.connectedComponents(sure_fg_1)
        markers_1 = markers_1 + 1
        markers_1[unknown_1 == 255] = 0
        markers_1 = cv2.watershed(image, markers_1)

        contours_1 = []
        for label in range(2, num_labels_1 + 2):
            mask_label = np.uint8(markers_1 == label)
            contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_1.extend([cnt for cnt in contours if cv2.contourArea(cnt) > 1000])

        # Method 2: Using Distance Transform with 0.8 threshold
        dist_transform_2 = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, dist_thresh_2 = cv2.threshold(dist_transform_2, 0.8 * dist_transform_2.max(), 255, 0)
        dist_thresh_2 = np.uint8(dist_thresh_2)
        sure_bg_2 = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
        sure_fg_2 = dist_thresh_2
        unknown_2 = cv2.subtract(sure_bg_2, sure_fg_2)
        _, markers_2 = cv2.connectedComponents(sure_fg_2)
        markers_2 = markers_2 + 1
        markers_2[unknown_2 == 255] = 0
        markers_2 = cv2.watershed(image, markers_2)

        contours_2 = []
        for marker in np.unique(markers_2):
            if marker <= 1:
                continue
            mask_label = np.zeros(markers_2.shape, dtype="uint8")
            mask_label[markers_2 == marker] = 255
            contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_2.extend([cnt for cnt in contours if cv2.contourArea(cnt) > 1000])

        # Combine contours from both methods
        combined_contours = contours_1 + contours_2 + filtered_contours_2

        # Draw combined contours
        output_image_combined = image.copy()
        cv2.drawContours(output_image_combined, combined_contours, -1, (0, 0, 255), 10)  # Red color
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        cv2.imwrite(os.path.join(output_folder, f'{file_name}_combined.jpg'), output_image_combined)

    print(f"Processed and saved: {file_name}")


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(input_folder, filename)
            process_image(file_path, output_folder)


def main():
    process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()