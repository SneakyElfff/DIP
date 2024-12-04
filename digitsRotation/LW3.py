import cv2
import numpy as np
import os

def isolate_numbers(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_border_colour = np.array([40, 140, 130])
    upper_border_colour = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_border_colour, upper_border_colour)

    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask


def rotate_numbers(numbers_image, mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    if not filtered_contours:
        raise ValueError("No contours detected.")

    result = np.zeros_like(numbers_image)

    for contour in filtered_contours:
        min_rect = cv2.minAreaRect(contour)
        center, size, angle = min_rect

        if size[0] > size[1]:
            angle += 90
            size = (size[1], size[0])

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv2.warpAffine(numbers_image, rotation_matrix, (numbers_image.shape[1], numbers_image.shape[0]))

        width, height = map(int, size)
        x, y = int(center[0] - width / 2), int(center[1] - height / 2)

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(rotated_image.shape[1], x + width), min(rotated_image.shape[0], y + height)

        result[y1:y2, x1:x2] = rotated_image[y1:y2, x1:x2]

    return result


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Ошибка: невозможно загрузить файл {filename}")
                continue

            try:
                numbers_image, mask = isolate_numbers(image)

                if(filename=='photo3_processed.jpg'):
                    result = rotate_numbers(numbers_image, mask, min_area=5000)
                else:
                    result = rotate_numbers(numbers_image, mask, min_area=15000)

                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_folder, f"{base_name}_processed.jpg")
                cv2.imwrite(output_path, result)
                print(f"Обработан файл: {filename}")
            except ValueError as e:
                print(f"Ошибка при обработке файла {filename}: {e}")


def main():
    input_folder = "/Users/nina/PycharmProjects/DIP/figures"
    output_folder = "/Users/nina/PycharmProjects/DIP/results"

    process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
