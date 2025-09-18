import os
import cv2
import numpy as np
from PIL import Image
from moonraker_interface import start_print, poll_print_status, capture_image, cancel_print
from preprocess_image import find_target_contour, crop_from_contour, auto_canny
from file_handler import send_telegram_image


class Model:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.load_config()

    def _convert_cv2_to_pil(self, cv2_image):
        if cv2_image is None:
            return None
        if len(cv2_image.shape) == 2:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB)
        else:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_image)

    def load_config(self):
        with open('config.txt', 'r') as config:
            self.PRINTER_ADDRESS = config.readline().strip()
            self.GCODE_FILE = config.readline().strip()
            self.TOKEN = config.readline().strip()
            self.CHAT_ID = config.readline().strip()

    def start_print(self):
        start_print(self.PRINTER_ADDRESS, self.GCODE_FILE)

    def poll_print_progress(self):
        return poll_print_status(self.PRINTER_ADDRESS)

    def cancel_print(self):
        cancel_print(self.PRINTER_ADDRESS)

    def capture_and_load_image(self, path):
        capture_image(self.PRINTER_ADDRESS, path)
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            raise ValueError(f"Failed to read image from {path}. The file may be invalid or missing.")
        self.processed_image = self.original_image.copy()
        return self._convert_cv2_to_pil(self.original_image)

    def process_image_step(self, step_name, sigma, debug_mode=False):
        if self.original_image is None and step_name != 'Original':
            raise ValueError("An image must be captured first.")
        if self.processed_image is None and step_name != 'Original':
            self.processed_image = self.original_image.copy()

        current_image = self.processed_image

        if step_name == 'Original':
            self.processed_image = self.original_image.copy()
        elif step_name == 'Grayscale':
            if len(current_image.shape) == 3:
                self.processed_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        elif step_name == 'Blurred':
            gray_input = current_image
            if len(current_image.shape) == 3:
                gray_input = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.GaussianBlur(gray_input, (7, 7), 0)
        elif step_name == 'Find Outer Edges':
            gray_input = current_image
            if len(current_image.shape) == 3:
                gray_input = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            edges = auto_canny(gray_input, sigma)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            output_image = np.zeros_like(edges)
            cv2.drawContours(output_image, contours, -1, 255, 1)
            self.processed_image = output_image
        elif step_name == 'Closed Edges':
            gray_input = current_image
            if len(current_image.shape) == 3:
                raise TypeError("Input for 'Closed Edges' must be a binary edge image, not a color image.")
            kernel = np.ones((9, 9), np.uint8)
            self.processed_image = cv2.morphologyEx(gray_input, cv2.MORPH_CLOSE, kernel)
        elif step_name == 'Crop to Shape':
            contour = find_target_contour(current_image, debug_mode=debug_mode)
            self.processed_image = crop_from_contour(self.original_image, contour)
        else:
            raise ValueError(f"Unknown processing step: {step_name}")

        return self._convert_cv2_to_pil(self.processed_image)

    def run_full_pipeline(self, sigma, debug_mode=False):
        if self.original_image is None:
            raise ValueError("Cannot run pipeline, an image must be captured first.")

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = auto_canny(blurred, sigma)
        contours_img = np.zeros_like(edges)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, -1, 255, 1)
        kernel = np.ones((9, 9), np.uint8)
        closed = cv2.morphologyEx(contours_img, cv2.MORPH_CLOSE, kernel)
        target_contour = find_target_contour(closed, debug_mode=debug_mode)
        final_image = crop_from_contour(self.original_image, target_contour)
        self.processed_image = final_image

        return self._convert_cv2_to_pil(self.processed_image)

    def save_temp_image(self, path):
        if self.processed_image is None:
            raise ValueError("There is no processed image to save.")
        cv2.imwrite(path, self.processed_image)

    def save_image(self, classification):
        if self.processed_image is None:
            raise ValueError("There is no processed image to save.")

        image_save_dir = os.path.join("images", classification)
        os.makedirs(image_save_dir, exist_ok=True)

        try:
            existing_files = [f for f in os.listdir(image_save_dir) if f.endswith('.jpg')]
            existing_numbers = {int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()}
        except (IOError, ValueError):
            existing_numbers = set()

        next_available_number = 0
        while next_available_number in existing_numbers:
            next_available_number += 1

        save_path = os.path.join(image_save_dir, f"{next_available_number}.jpg")
        cv2.imwrite(save_path, self.processed_image)
        return save_path

    def send_telegram_notification(self, image_path, caption=""):
        send_telegram_image(self.TOKEN, self.CHAT_ID, image_path, caption)