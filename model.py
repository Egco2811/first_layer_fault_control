import os
import cv2
import numpy as np
import json
from PIL import Image
from config import Config
from moonraker_interface import (start_print, poll_print_status, capture_image, cancel_print,
                                 adjust_z_offset, restart_firmware, auto_home, send_gcode)
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
        self.last_successful_corners = None
        self.last_found_corners = None
        self.load_last_successful_corners()

    def has_image(self):
        """Safe check if an image is currently loaded."""
        return self.original_image is not None

    def _convert_cv2_to_pil(self, cv2_image):
        if cv2_image is None: return None
        if len(cv2_image.shape) == 2:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2RGB)
        else:
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cv2_image)

    def load_last_successful_corners(self):
        if os.path.exists(LAST_SUCCESSFUL_PARAMS_FILE):
            with open(LAST_SUCCESSFUL_PARAMS_FILE, 'r') as f:
                try:
                    data = json.load(f)
                    self.last_successful_corners = np.array(data['corners'], dtype=np.int32)
                except (json.JSONDecodeError, KeyError):
                    self.last_successful_corners = None

    def save_last_successful_corners(self, corners):
        data = {'corners': corners.tolist()}
        with open(LAST_SUCCESSFUL_PARAMS_FILE, 'w') as f:
            json.dump(data, f)
        self.last_successful_corners = corners

    def commit_last_found_corners(self):
        if self.last_found_corners is not None:
            self.save_last_successful_corners(self.last_found_corners)

    def start_print(self):
        start_print(Config.PRINTER_URL, Config.GCODE_FILE)

    def poll_print_progress(self):
        return poll_print_status(Config.PRINTER_URL)

    def cancel_print(self):
        cancel_print(Config.PRINTER_URL)

    def capture_and_load_image(self, path):
        if capture_image(Config.PRINTER_URL, path):
            self.original_image = cv2.imread(path)
            if self.original_image is None:
                raise ValueError("Failed to read captured image from disk.")
            
            self.pipeline_cache.clear()
            self.pipeline_cache['Original'] = self.original_image.copy()
            self.processed_image = self.original_image.copy()
            self.last_found_corners = None
            return self._convert_cv2_to_pil(self.original_image)
        raise ValueError("Failed to communicate with printer webcam.")

    def invalidate_sigma_dependent_cache(self):
        for step in ["Find Outer Edges", "Closed Edges", "Crop to Shape"]:
            self.pipeline_cache.pop(step, None)

    def _ensure_step_in_cache(self, step_name, sigma, debug_mode):
        if step_name in self.pipeline_cache: return

        if self.original_image is None:
            raise RuntimeError("No image captured. Please capture an image first.")

        idx = self.PIPELINE_STEPS.index(step_name)
        if idx == 0:
            if 'Original' not in self.pipeline_cache:
                self.pipeline_cache['Original'] = self.original_image.copy()
            return

        prev_name = self.PIPELINE_STEPS[idx - 1]
        self._ensure_step_in_cache(prev_name, sigma, debug_mode)
        prev_img = self.pipeline_cache[prev_name]

        if step_name == 'Grayscale':
            processed = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY) if len(prev_img.shape) == 3 else prev_img
        elif step_name == 'Blurred':
            processed = cv2.GaussianBlur(prev_img, (7, 7), 0)
        elif step_name == 'Find Outer Edges':
            processed = auto_canny(prev_img, sigma)
        elif step_name == 'Closed Edges':
            kernel = np.ones((9, 9), np.uint8)
            processed = cv2.morphologyEx(prev_img, cv2.MORPH_CLOSE, kernel)
        elif step_name == 'Crop to Shape':
            clean = self.pipeline_cache['Closed Edges']
            contour = find_target_contour(clean, debug_mode=debug_mode)
            if contour is None:
                raise RuntimeError("Could not find a rectangular shape in the image.")
            self.last_found_corners = contour
            processed = crop_from_contour(self.pipeline_cache['Original'], contour)
        else:
            raise ValueError(f"Unknown processing step: {step_name}")

        self.pipeline_cache[step_name] = processed

    def process_image_step(self, target_step_name, sigma, debug_mode=False):
        if not self.has_image():
            raise RuntimeError("No image to process.")
        
        self._ensure_step_in_cache(target_step_name, sigma, debug_mode)
        self.processed_image = self.pipeline_cache[target_step_name]
        return self._convert_cv2_to_pil(self.processed_image)

    def run_full_pipeline_with_retry(self, user_sigma, debug_mode=False):
        if not self.has_image():
            raise RuntimeError("No image to process.")

        user_sigma = round(user_sigma, 2)
        lower, upper = max(0.0, user_sigma - 0.3), min(1.0, user_sigma + 0.3)
        sigmas = [user_sigma] + [round(lower + i * 0.01, 2) for i in range(int((upper - lower) / 0.01))]
        
        for s in sigmas:
            try:
                self.invalidate_sigma_dependent_cache()
                return self.process_image_step('Crop to Shape', s, debug_mode), s, False
            except RuntimeError:
                continue
        
        if self.last_successful_corners is not None:
            return self.crop_with_fallback(), user_sigma, True
        
        raise RuntimeError("Failed to find shape and no fallback parameters available.")

    def crop_with_fallback(self):
        if not self.has_image():
            raise RuntimeError("No image to process.")
        if self.last_successful_corners is None:
            raise RuntimeError("No fallback parameters saved.")
        if 'Original' not in self.pipeline_cache:
             self.pipeline_cache['Original'] = self.original_image.copy()
        
        cropped = crop_from_contour(self.pipeline_cache['Original'], self.last_successful_corners)
        self.processed_image = cropped
        return self._convert_cv2_to_pil(cropped)

    def save_temp_image(self, path):
        if self.processed_image is not None:
            cv2.imwrite(path, self.processed_image)

    def save_image(self, classification):
        if self.processed_image is None: raise ValueError("No processed image to save.")
        dir_path = os.path.join("images", classification)
        os.makedirs(dir_path, exist_ok=True)
        
        existing = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
        nums = {int(f.split('.')[0]) for f in existing if f.split('.')[0].isdigit()}
        idx = 0
        while idx in nums: idx += 1
        
        path = os.path.join(dir_path, f"{idx}.jpg")
        cv2.imwrite(path, self.processed_image)
        return path

    def send_telegram_notification(self, image_path, caption=""):
        if Config.TELEGRAM_TOKEN:
            send_telegram_image(Config.TELEGRAM_TOKEN, Config.TELEGRAM_CHAT_ID, image_path, caption)

    def adjust_z_offset(self, adj): adjust_z_offset(Config.PRINTER_URL, adj)
    def restart_firmware(self): restart_firmware(Config.PRINTER_URL)
    def auto_home(self): auto_home(Config.PRINTER_URL)
    def send_gcode(self, cmd): send_gcode(Config.PRINTER_URL, cmd)
    def get_webcam_stream_url(self): return Config.get_webcam_url()