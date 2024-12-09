import os
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass
import torchvision
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Normalize

from LW4 import ModelArchitecture, device


def load_model():
    model = ModelArchitecture().to(device)
    model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location=device))
    model.eval()
    return model


def get_transform():
    return Compose([
        ToPILImage(),
        Resize((28, 28)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])


def find_contours(image):
    height, width = image.shape[:2]
    default_width, default_height = 4624, 2080
    scale_factor = (width * height) / (default_width * default_height)
    min_num_area = int(15000 * scale_factor)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_border_colour = np.array([40, 140, 130])
    upper_border_colour = np.array([90, 255, 255])
    mask = cv2.inRange(hsv_image, lower_border_colour, upper_border_colour)

    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_num_area]

    return mask_cleaned, contours


def compute_shift(img):
    cy, cx = center_of_mass(img)
    rows, cols = img.shape
    return np.round(cols / 2.0 - cx).astype(int), np.round(rows / 2.0 - cy).astype(int)


def shift_image(img, shiftx, shifty):
    rows, cols = img.shape
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    return cv2.warpAffine(img, M, (cols, rows))


def convert_to_mnist(mask, contour):
    x, y, w, h = cv2.boundingRect(contour)
    digit_image = mask[y:y + h, x:x + w]
    scale = min(20 / digit_image.shape[0], 20 / digit_image.shape[1])
    new_w, new_h = int(digit_image.shape[1] * scale), int(digit_image.shape[0] * scale)
    resized_digit = cv2.resize(digit_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    centered_digit = np.zeros((20, 20), dtype=np.uint8)
    centered_digit[(20 - new_h) // 2:(20 - new_h) // 2 + new_h,
                   (20 - new_w) // 2:(20 - new_w) // 2 + new_w] = resized_digit

    padded_digit = np.pad(centered_digit, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    shiftx, shifty = compute_shift(padded_digit)
    return shift_image(padded_digit, shiftx, shifty)


def predict(image, model, transform):
    mask, contours = find_contours(image)
    predictions = []

    for cnt in contours:
        digit_image = convert_to_mnist(mask, cnt)
        digit_image_tensor = transform(digit_image).unsqueeze(0).to(device)
        rotated_image_tensor = torchvision.transforms.functional.rotate(digit_image_tensor, angle=180)

        with torch.no_grad():
            result = model(digit_image_tensor)
            probability = torch.softmax(result, dim=1)
            max_prob, predicted = probability.max(dim=1)

            result_rotated = model(rotated_image_tensor)
            probability_rotated = torch.softmax(result_rotated, dim=1)
            max_prob_rotated, predicted_rotated = probability_rotated.max(dim=1)

        if max_prob > max_prob_rotated:
            predictions.append(predicted.item())
        else:
            predictions.append(predicted_rotated.item())

    return predictions


def process_folder(folder, model, transform):
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found.")
        return

    print(f"Folder '{folder}':")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load the file: {filename}")
            continue

        predictions = predict(image, model, transform)
        print(f"{filename}: Digits recognised (top to bottom): {list(reversed(predictions))}")


if __name__ == "__main__":
    model = load_model()
    transform = get_transform()

    folder_to_process = "/Users/nina/PycharmProjects/DIP/results"
    process_folder(folder_to_process, model, transform)