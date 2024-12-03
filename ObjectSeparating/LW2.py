import cv2
import numpy as np
import random
import os

def enhance_image(image):
    """Увеличение насыщенности и контраста изображения."""
    # Преобразование в HSV для увеличения насыщенности
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_scale = 2.5  # Коэффициент увеличения насыщенности
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255)
    image_with_boosted_saturation = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Преобразование в YUV для повышения контраста
    yuv_image = cv2.cvtColor(image_with_boosted_saturation, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])  # Применение CLAHE к яркости
    enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return enhanced_image

def process_image(image_path, output_dir):
    """Обработка одного изображения."""
    image = cv2.imread(image_path)

    # Enhance the image (increase saturation and contrast)
    image = enhance_image(image)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for blue
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a mask for blue regions
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Morphological operation to clean the mask
    kernel = np.ones((19, 19), np.uint8)
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
        _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
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
    output_filename = os.path.basename(image_path).replace('.jpg', '_processed.jpg')
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, output_image)
    print(f"Processed: {image_path}, saved to {output_path}")

# Define the input and output directories
input_dir = "/Users/nina/PycharmProjects/DIP/figures"
output_dir = "/Users/nina/PycharmProjects/DIP/results"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        process_image(image_path, output_dir)