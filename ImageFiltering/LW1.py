import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def load_image(file_path):
    return cv2.imread(file_path)


def isolate_objects(image):
    # конвертировать в HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_border_colour = np.array([90, 50, 50])
    upper_border_colour = np.array([240, 255, 255])

    mask = cv2.inRange(hsv, lower_border_colour, upper_border_colour)

    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask


def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours(contours, min_area=1000):
    filtered_contours = []

    for i in contours:
        area = cv2.contourArea(i)

        if area > min_area:
            # ограничить зону поиска
            x, y, w, h = cv2.boundingRect(i)
            aspect_ratio = w / float(h)

            if 0.8 <= aspect_ratio <= 1.4:
                filtered_contours.append(i)

    return filtered_contours


def create_mask_from_contours(image_shape, contours):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # на основе w и h

    cv2.drawContours(mask, contours, -1, (255), -1)

    return mask


def draw_contours(image, contours):
    # для улучшения внешнего вида контуров
    result_with_contours = image.copy()

    result_with_contours = cv2.GaussianBlur(result_with_contours, (5, 5), 0)

    cv2.drawContours(result_with_contours, contours, -1, (0, 255, 255), 10)

    return result_with_contours


def edit_image(file_path, output_dir):
    original_image = load_image(file_path)

    objects, colour_mask = isolate_objects(original_image)

    contours = find_contours(colour_mask)
    large_contours = filter_contours(contours)

    final_mask = create_mask_from_contours(original_image.shape, large_contours)
    result_image = cv2.bitwise_and(original_image, original_image, mask=final_mask)

    result_with_contours = draw_contours(result_image, large_contours)

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    cv2.imwrite(os.path.join(output_dir, f'{file_name}_processed.jpg'), result_with_contours)

    return original_image, result_with_contours


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

            original, edited = edit_image(file_path, output_folder)
            display_images(original, edited, f"Обработка изображения: {filename}")

            print(f"Обработан файл: {filename}")


def main():
    input_folder = '/Users/nina/PycharmProjects/DIP/ImageFiltering/figures'
    output_folder = '/Users/nina/PycharmProjects/DIP/ImageFiltering/results'

    process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()