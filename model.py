import os
import cv2
import numpy as np
import json
from PIL import Image
from moonraker_interface import start_print, poll_print_status, capture_image, cancel_print
from preprocess_image import find_target_contour, crop_from_contour, auto_canny
from file_handler import send_telegram_image

LAST_SUCCESSFUL_PARAMS_FILE = "last_successful_params.json"

class Model:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.pipeline_cache = {}
        self.PIPELINE_STEPS = [
            "Original", "Grayscale", "Blurred",
            "Find Outer Edges", "Closed Edges", "Crop to Shape"
        ]
        self.load_config()
        self.last_successful_corners = None
        self.last_found_corners = None
        self.load_last_successful_corners()

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

    def load_last_successful_corners(self):
        if os.path.exists(LAST_SUCCESSFUL_PARAMS_FILE):
            with open(LAST_SUCCESSFUL_PARAMS_FILE, 'r') as f:
                try:
                    data = json.load(f)
                    self.last_successful_corners = np.array(data['corners'], dtype=np.int32)
                    print("Loaded last successful crop parameters.")
                except (json.JSONDecodeError, KeyError):
                    print("Could not load last successful crop parameters from file.")
                    self.last_successful_corners = None

    def save_last_successful_corners(self, corners):
        data = {'corners': corners.tolist()}
        with open(LAST_SUCCESSFUL_PARAMS_FILE, 'w') as f:
            json.dump(data, f)
        self.last_successful_corners = corners
        print("Saved new successful crop parameters.")

    def commit_last_found_corners(self):
        if self.last_found_corners is not None:
            self.save_last_successful_corners(self.last_found_corners)
            print("Committed last found corners as new successful parameters.")
        else:
            print("Commit requested, but no new corners were found in this session.")

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

        self.pipeline_cache.clear()
        self.pipeline_cache['Original'] = self.original_image.copy()
        self.processed_image = self.original_image.copy()
        self.last_found_corners = None

        return self._convert_cv2_to_pil(self.original_image)

    def invalidate_sigma_dependent_cache(self):
        sigma_dependent_steps = ["Find Outer Edges", "Closed Edges", "Crop to Shape"]
        for step in sigma_dependent_steps:
            if step in self.pipeline_cache:
                self.pipeline_cache.pop(step)
        print("Sigma-dependent cache steps cleared.")

    def _ensure_step_in_cache(self, step_name, sigma, debug_mode):
        if step_name in self.pipeline_cache:
            return

        try:
            step_index = self.PIPELINE_STEPS.index(step_name)
        except ValueError:
            raise ValueError(f"Unknown processing step: {step_name}")

        if step_index == 0:
            if 'Original' not in self.pipeline_cache:
                raise RuntimeError("Original image not found in cache. Please capture an image first.")
            return

        prev_step_name = self.PIPELINE_STEPS[step_index - 1]
        self._ensure_step_in_cache(prev_step_name, sigma, debug_mode)

        prev_image = self.pipeline_cache[prev_step_name]

        if step_name == 'Grayscale':
            if len(prev_image.shape) == 3:
                processed = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
            else:
                processed = prev_image
        elif step_name == 'Blurred':
            processed = cv2.GaussianBlur(prev_image, (7, 7), 0)
        elif step_name == 'Find Outer Edges':
            edges = auto_canny(prev_image, sigma)
            contours_img = np.zeros_like(edges)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contours_img, contours, -1, 255, 1)
            processed = contours_img
        elif step_name == 'Closed Edges':
            kernel = np.ones((9, 9), np.uint8)
            processed = cv2.morphologyEx(prev_image, cv2.MORPH_CLOSE, kernel)
        elif step_name == 'Crop to Shape':
            clean_outline_image = self.pipeline_cache['Closed Edges']
            contour = find_target_contour(clean_outline_image, debug_mode=debug_mode)
            self.last_found_corners = contour
            original_color_image = self.pipeline_cache['Original']
            processed = crop_from_contour(original_color_image, contour)
        else:
            raise ValueError(f"Execution logic for step '{step_name}' not defined.")

        self.pipeline_cache[step_name] = processed

    def process_image_step(self, target_step_name, sigma, debug_mode=False):
        if self.original_image is None:
            raise ValueError("An image must be captured first.")

        self._ensure_step_in_cache(target_step_name, sigma, debug_mode)
        self.processed_image = self.pipeline_cache[target_step_name]
        return self._convert_cv2_to_pil(self.processed_image)

    def run_full_pipeline(self, sigma, debug_mode=False):
        return self.process_image_step('Crop to Shape', sigma, debug_mode)

    def run_full_pipeline_with_retry(self, user_sigma, debug_mode=False):
        user_sigma = round(user_sigma, 2)
        lower_bound = max(0.0, user_sigma - 0.3)
        upper_bound = min(1.0, user_sigma + 0.3)
        step = 0.01
        num_steps = int(round((upper_bound - lower_bound) / step))
        sigma_list = [round(lower_bound + i * step, 2) for i in range(num_steps + 1)]

        try:
            sigma_list.remove(user_sigma)
        except ValueError:
            pass

        sigma_values_to_try = [user_sigma] + sigma_list

        for sigma in sigma_values_to_try:
            try:
                self.invalidate_sigma_dependent_cache()
                print(f"Attempting pipeline with sigma={sigma:.2f}")
                final_image = self.run_full_pipeline(sigma, debug_mode)
                print(f"Pipeline succeeded with sigma={sigma:.2f}")
                return final_image, sigma, False
            except RuntimeError as e:
                print(f"Pipeline failed for sigma={sigma:.2f}: {e}")

        print("All sigma retries failed. Attempting to use last successful crop parameters.")
        if self.last_successful_corners is not None:
            try:
                self._ensure_step_in_cache('Closed Edges', user_sigma, debug_mode)
                original_image = self.pipeline_cache['Original']
                cropped_image = crop_from_contour(original_image, self.last_successful_corners)
                self.processed_image = cropped_image
                print("Successfully cropped image using fallback parameters.")
                return self._convert_cv2_to_pil(cropped_image), user_sigma, True
            except Exception as e:
                 print(f"Fallback cropping failed: {e}")
                 raise RuntimeError("Failed to process image after trying multiple sigma values and the fallback also failed.")
        else:
            raise RuntimeError("Failed to process image after trying multiple sigma values and no fallback is available.")


    def crop_with_fallback(self):
        if self.last_successful_corners is not None:
            if 'Original' not in self.pipeline_cache:
                raise RuntimeError("Original image is not in cache for fallback cropping.")
            original_image = self.pipeline_cache['Original']
            cropped_image = crop_from_contour(original_image, self.last_successful_corners)
            self.processed_image = cropped_image
            return self._convert_cv2_to_pil(cropped_image)
        else:
            raise RuntimeError("No fallback parameters available.")

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