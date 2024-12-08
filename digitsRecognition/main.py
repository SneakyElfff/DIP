import os
import cv2
import torch
from scipy.ndimage import center_of_mass
import torchvision
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Normalize
import numpy as np

from LW4 import MyModel, device

# Загрузка сохранённой модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

# Преобразования для предобработки изображений
transform = Compose([
    ToPILImage(),
    Resize((28, 28)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

def prepare_contours(image):
    height, width = image.shape[:2]
    default_width, default_height = 4624, 2080
    scale_factor = (width * height) / (default_width * default_height)
    min_num_area = int(15000 * scale_factor)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_salad = np.array([40, 140, 130])
    upper_salad = np.array([90, 255, 255])
    mask = cv2.inRange(hsv_image, lower_salad, upper_salad)

    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_num_area]

    return mask_cleaned, contours

def get_best_shift(img):
    cy, cx = center_of_mass(img)
    rows, cols = img.shape
    return np.round(cols / 2.0 - cx).astype(int), np.round(rows / 2.0 - cy).astype(int)

def shift_image(img, shiftx, shifty):
    rows, cols = img.shape
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    return cv2.warpAffine(img, M, (cols, rows))

def contour_to_mnist(mask, contour):
    x, y, w, h = cv2.boundingRect(contour)
    digit_image = mask[y:y + h, x:x + w]
    scale = min(20 / digit_image.shape[0], 20 / digit_image.shape[1])
    new_w, new_h = int(digit_image.shape[1] * scale), int(digit_image.shape[0] * scale)
    resized_digit = cv2.resize(digit_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    centered_digit = np.zeros((20, 20), dtype=np.uint8)
    centered_digit[(20 - new_h) // 2:(20 - new_h) // 2 + new_h,
                   (20 - new_w) // 2:(20 - new_w) // 2 + new_w] = resized_digit

    padded_digit = np.pad(centered_digit, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    shiftx, shifty = get_best_shift(padded_digit)
    return shift_image(padded_digit, shiftx, shifty)

def predict_label(image):
    mask, contours = prepare_contours(image)
    predictions = []
    for cnt in contours:
        digit_image = contour_to_mnist(mask, cnt)
        digit_image_tensor = transform(digit_image).unsqueeze(0).to(device)
        rotated_image_tensor = torchvision.transforms.functional.rotate(digit_image_tensor, angle=180)

        with torch.no_grad():
            output_original = model(digit_image_tensor)
            probabilities_original = torch.softmax(output_original, dim=1)
            max_prob_original, predicted_original = probabilities_original.max(dim=1)

            output_rotated = model(rotated_image_tensor)
            probabilities_rotated = torch.softmax(output_rotated, dim=1)
            max_prob_rotated, predicted_rotated = probabilities_rotated.max(dim=1)

        if max_prob_original > max_prob_rotated:
            predictions.append(predicted_original.item())
        else:
            predictions.append(predicted_rotated.item())

    return predictions

def process_directory(directory):
    if not os.path.exists(directory):
        print(f"Каталог '{directory}' не найден.")
        return

    print("Результаты для изображений в каталоге:")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        image = cv2.imread(file_path)
        if image is None:
            print(f"Не удалось загрузить файл: {filename}")
            continue
        predictions = predict_label(image)
        print(f"{filename}: Найдены цифры (сверху вниз): {list(reversed(predictions))}")

# Укажите каталог с изображениями
directory_to_process = "./images"
process_directory(directory_to_process)