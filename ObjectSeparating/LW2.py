import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

from ImageFiltering.LW1 import display_images


def enhance_image(image):
    """Increase saturation and contrast of the image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_scale = 2.5
    # ограничить значения матрицы насыщенности до допустимых пределов
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255)
    image_with_boosted_saturation = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Y - яркость
    yuv_image = cv2.cvtColor(image_with_boosted_saturation, cv2.COLOR_BGR2YUV)
    # улучшить локальный контраст изображения
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])
    enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return enhanced_image


def calculate_area(mask):
    """Calculate the area of the object represented by the mask."""
    return np.sum(mask) / 255


def calculate_perimeter(mask):
    """Calculate the perimeter of the object represented by the mask."""
    # стирание пикселей с краев
    kernel = np.ones((3, 3), np.uint8)
    # если хотя бы один пиксель из соседей фона (0), центральный пиксель также становится фоном
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    boundary = mask - eroded_mask
    perimeter = np.sum(boundary) / 255
    return perimeter


def calculate_aspect_ratio(mask):
    """
    Calculate the aspect ratio (width/height) of the object represented by the mask.
    """
    # координаты всех белых пикселей (объектов)
    y_coords, x_coords = np.where(mask > 0)

    # сли объект отсутствует
    if len(x_coords) == 0 or len(y_coords) == 0:
        return 0

    # границы объекта
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    aspect_ratio = width / height
    return aspect_ratio


def preprocess_image(image_path):
    """Load, enhance the image, and prepare the cleaned mask."""
    image = cv2.imread(image_path)
    enhanced_image = enhance_image(image)

    hsv_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)

    lower_border_colour = np.array([100, 100, 50])
    upper_border_colour = np.array([140, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_border_colour, upper_border_colour)

    kernel = np.ones((19, 19), np.uint8)
    # дилатация -> эрозия, заполнение "дыр"
    cleaned_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    return enhanced_image, cleaned_mask


def generate_markers_and_features(cleaned_mask, output_image):
    """Generate markers using watershed and calculate object features."""
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_features = []  # [area, perimeter, aspect_ratio]
    object_masks = []
    object_contours = []

    object_id = 1

    for i, contour in enumerate(contours):
        group_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(group_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # расстояние до ближайшего фона
        dist_transform = cv2.distanceTransform(group_mask, cv2.DIST_L2, 5)
        # помечаются центры объектов
        _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # найти неизвестные области (границ перекрывающихся объектов)
        unknown = cv2.subtract(group_mask, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # фон - 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(output_image, markers)

        for marker_id in range(2, np.max(markers) + 1):
            object_mask = np.zeros_like(markers, dtype=np.uint8)
            object_mask[markers == marker_id] = 255

            area = calculate_area(object_mask)
            perimeter = calculate_perimeter(object_mask)
            aspect_ratio = calculate_aspect_ratio(object_mask)

            print(f"Object: {object_id}, Area: {area}, Perimeter: {perimeter}, Aspect Ratio: {aspect_ratio:.2f}")

            object_features.append([area, perimeter, aspect_ratio])
            object_masks.append(object_mask)
            # object_contours.append(contour)

            object_id += 1

    return object_features, object_masks


def cluster_objects_and_recolor(output_image, object_features, object_masks, image_path, output_dir):
    """Cluster the objects and apply colors based on clustering results."""
    n_clusters = 3

    if len(object_features) > 0:
        effective_clusters = min(n_clusters, len(object_features))
        # результат алгоритма одинаковый при каждом запуске
        kmeans = KMeans(n_clusters=effective_clusters, random_state=0)
        labels = kmeans.fit_predict(object_features)
    else:
        print(f"No objects found in {image_path}")
        return

    class_colors = {
        0: (0, 0, 255),   # Class 1: Red
        1: (255, 0, 0),   # Class 2: Blue
        2: (0, 255, 0)    # Class 3: Green
    }

    for mask, label in zip(object_masks, labels):
        color = class_colors[label % n_clusters]  # Handle fewer clusters
        output_image[mask > 0] = color

    output_filename = os.path.basename(image_path).replace('.jpg', '_clustered.jpg')
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, output_image)
    print(f"Processed with clustering: {image_path}, saved to {output_path}")


def process_image(image_path, output_dir):
    """Process a single image by integrating the helper methods."""
    image, cleaned_mask = preprocess_image(image_path)

    object_features, object_masks = generate_markers_and_features(cleaned_mask, image)

    clustered_image = image.copy()
    cluster_objects_and_recolor(clustered_image, object_features, object_masks, image_path, output_dir)
    display_images(image, clustered_image, f"Кластеризация объектов изображения {image_path}")


def process_directory(input_dir, output_dir):
    """Process all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            process_image(image_path, output_dir)


def main():
    input_dir = "/Users/nina/PycharmProjects/DIP/figures"
    output_dir = "/Users/nina/PycharmProjects/DIP/results"

    process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()