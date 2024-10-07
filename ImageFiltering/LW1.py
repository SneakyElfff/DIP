import cv2
import numpy as np


def load_image(file_path):
    return cv2.imread(file_path)


def isolate_blue_objects(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result, mask


def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours(contours, min_area=1000):
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Approximate the contour to detect its shape
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

            # Bounding box to check aspect ratio and size consistency
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Filter by area and aspect ratio (between roughly 0.8 and 1.2 for squares and regular polygons)
            if 0.8 <= aspect_ratio <= 1.4:
                filtered_contours.append(cnt)
    return filtered_contours


def create_mask_from_contours(image_shape, contours):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255), -1)
    return mask


def process_image(file_path):
    # Load the image
    original = load_image(file_path)

    # Isolate blue objects
    blue_objects, blue_mask = isolate_blue_objects(original)

    # Find contours
    contours = find_contours(blue_mask)

    # Filter contours by area and bounding box aspect ratio
    large_contours = filter_contours(contours)

    # Create a mask from filtered contours
    final_mask = create_mask_from_contours(original.shape, large_contours)

    # Apply the final mask to the original image
    result = cv2.bitwise_and(original, original, mask=final_mask)

    # Save results
    cv2.imwrite('/Users/nina/PycharmProjects/DIP/ImageFiltering/results/blue_objects_filtered.jpg', blue_objects)
    cv2.imwrite('/Users/nina/PycharmProjects/DIP/ImageFiltering/results/final_result_filtered.jpg', result)


# Usage
process_image('/Users/nina/PycharmProjects/DIP/ImageFiltering/figures/1725544579789.jpg')
