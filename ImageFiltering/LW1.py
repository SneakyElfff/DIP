import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


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


def process_image(file_path, output_dir):
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

    # **Draw contours on the processed image**:
    # This will outline the detected blue objects
    result_with_contours = result.copy()

    # Размытие изображения для улучшения внешнего вида контуров
    result_with_contours = cv2.GaussianBlur(result_with_contours, (5, 5), 0)

    cv2.drawContours(result_with_contours, large_contours, -1, (0, 255, 0), 10)

    # Save final result with contours
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_processed.jpg'), result_with_contours)

    return original, result_with_contours


def display_images(original, processed, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Оригинал')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title('Обработанное с контурами')
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(input_folder, filename)
            original, processed = process_image(file_path, output_folder)
            display_images(original, processed, f"Обработка изображения: {filename}")
            print(f"Обработан файл: {filename}")


input_folder = '/Users/nina/PycharmProjects/DIP/ImageFiltering/figures'
output_folder = '/Users/nina/PycharmProjects/DIP/ImageFiltering/results'
process_folder(input_folder, output_folder)