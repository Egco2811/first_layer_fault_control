import threading
import time
import os
import sys
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from predictor import Predictor
from classifier import train_model, generate_confusion_matrix_data
from config import Config

UNPROCESSED_IMAGE_FILE = "unprocessed.jpg"
TEMP_IMAGE_FILE = "temp_autonomous_capture.jpg"

class StderrSuppressor:
    def __init__(self):
        self._original_stderr_fd = None
        self._devnull_fd = None

    def __enter__(self):
        sys.stderr.flush()
        self._original_stderr_fd = os.dup(2)
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull_fd, 2)

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self._original_stderr_fd, 2)
        os.close(self._devnull_fd)
        os.close(self._original_stderr_fd)

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.predictor = Predictor()
        self.autonomous_running = False
        self.webcam_running = False
        self._stop_training_flag = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.start_webcam_stream()

    def _run_task(self, target_func, *args):
        self.executor.submit(self._safe_execute, target_func, *args)

    def _safe_execute(self, func, *args):
        try:
            func(*args)
        except Exception as e:
            self.view.after(0, lambda: self.view.show_error("Execution Error", str(e)))

    def on_close(self):
        self.stop_webcam_stream()
        self.executor.shutdown(wait=False)
        self.view.destroy()

    def print_shape(self):
        self._run_task(self._print_shape_task)

    def _print_shape_task(self):
        self.view.after(0, lambda: self.view.set_ui_state('BUSY'))
        self.view.after(0, lambda: self.view.update_status("Starting print..."))
        try:
            self.model.start_print()
            print_succeeded = False
            for state, message in self.model.poll_print_progress():
                self.view.after(0, lambda m=message: self.view.update_status(m))
                if state == "complete":
                    print_succeeded = True
                    break
                elif state in ["error", "cancelled"]:
                    break
            final_msg = "Print finished." if print_succeeded else "Print failed or was cancelled."
            self.view.after(0, lambda: self.view.update_status(final_msg))
        finally:
            self.view.after(0, lambda: self.view.set_ui_state('IDLE'))

    def capture_and_display_image(self):
        self._run_task(self._capture_task)

    def _capture_task(self):
        self.view.after(0, lambda: self.view.set_ui_state('BUSY'))
        self.view.after(0, lambda: self.view.update_status("Capturing image..."))
        try:
            image = self.model.capture_and_load_image(UNPROCESSED_IMAGE_FILE)
            self.view.after(0, lambda: self.view.update_image_display(image))
            self.view.after(0, lambda: self.view.update_status("Image captured. Ready to process."))
            self.view.after(0, lambda: self.view.set_ui_state('CAPTURED'))
        except Exception as e:
            self.view.after(0, lambda: self.view.set_ui_state('IDLE'))
            raise e

    def view_step(self, step_name):
        self._run_task(self._view_step_task, step_name)

    def _view_step_task(self, step_name):
        if not self.model.has_image():
            self.view.after(0, lambda: self.view.show_error("Action Denied", "Please capture an image first."))
            return

        self.view.after(0, lambda: self.view.update_status(f"Applying '{step_name}'..."))
        try:
            sigma = self.view.canny_sigma_var.get()
            debug_mode = self.view.debug_mode_var.get()
            processed_image = self.model.process_image_step(step_name, sigma, debug_mode)
            self.view.after(0, lambda: self.view.update_image_display(processed_image))

            if step_name == 'Crop to Shape':
                self.view.after(0, lambda: self.view.update_status(f"Applied '{step_name}'. Ready to save."))
                self.view.after(0, lambda: self.view.set_ui_state('PROCESSED'))
            else:
                self.view.after(0, lambda: self.view.update_status(f"Applied '{step_name}' filter."))
        except Exception as e:
            if step_name == 'Crop to Shape' and isinstance(e, RuntimeError):
                self.view.after(0, lambda: self.view.update_status("Shape detection failed. Trying fallback..."))
                try:
                    fallback_image = self.model.crop_with_fallback()
                    self.view.after(0, lambda: self.view.update_image_display(fallback_image))
                    self.view.after(0, lambda: self.view.update_status("Used last successful crop parameters."))
                    self.view.after(0, lambda: self.view.set_ui_state('PROCESSED'))
                    self.view.after(0, lambda: self.view.show_info("Notice", "Used fallback parameters."))
                except Exception as fallback_e:
                    raise RuntimeError(f"Fallback failed: {fallback_e}")
            else:
                raise e

    def process_to_final(self):
        self._run_task(self._process_to_final_task)

    def _process_to_final_task(self):
        if not self.model.has_image():
            self.view.after(0, lambda: self.view.show_error("Action Denied", "Please capture an image first."))
            return

        self.view.after(0, lambda: self.view.set_ui_state('BUSY'))
        self.view.after(0, lambda: self.view.update_status("Running full pipeline..."))
        try:
            user_sigma = self.view.canny_sigma_var.get()
            debug_mode = self.view.debug_mode_var.get()
            final_image, successful_sigma, used_fallback = self.model.run_full_pipeline_with_retry(user_sigma, debug_mode)
            self.view.after(0, lambda: self.view.update_image_display(final_image))
            msg = f"Pipeline succeeded with sigma={successful_sigma:.2f}."
            if used_fallback:
                msg = "Shape detection failed. Used last successful parameters."
                self.view.after(0, lambda: self.view.show_info("Notice", "Used fallback parameters."))
            self.view.after(0, lambda: self.view.update_status(msg))
            self.view.after(0, lambda: self.view.set_ui_state('PROCESSED'))
        except Exception as e:
            self.view.after(0, lambda: self.view.set_ui_state('CAPTURED'))
            raise e

    def save_final_image(self):
        self._run_task(self._save_image_task)

    def _save_image_task(self):
        if not self.model.has_image():
            self.view.after(0, lambda: self.view.show_error("Action Denied", "Please capture an image first."))
            return

        self.view.after(0, lambda: self.view.update_status("Saving image..."))
        classification = self.view.classification_var.get()
        save_path = self.model.save_image(classification)
        self.model.commit_last_found_corners()
        self.view.after(0, lambda: self.view.show_info("Success", f"Saved to {save_path}"))
        self.view.after(0, lambda: self.view.update_status(f"Saved to '{classification}'."))

    def on_sigma_change(self, value):
        self.model.invalidate_sigma_dependent_cache()
        self.view.update_status(f"Sigma set to {float(value):.2f}. Cache cleared.")

    def toggle_autonomous_mode(self):
        if not self.autonomous_running:
            self.autonomous_running = True
            self.view.set_ui_state('AUTONOMOUS')
            threading.Thread(target=self.autonomous_loop, daemon=True).start()
        else:
            self.autonomous_running = False
            self.view.update_status("Stopping autonomous mode...")
            self._run_task(self.model.cancel_print)

    def autonomous_loop(self):
        while self.autonomous_running:
            try:
                self.view.after(0, lambda: self.view.update_status("[Auto] Starting print..."))
                self.model.start_print()
                print_succeeded = False
                for state, message in self.model.poll_print_progress():
                    if not self.autonomous_running: break
                    self.view.after(0, lambda m=message: self.view.update_status(f"[Auto] {m}"))
                    if state == "complete":
                        print_succeeded = True
                        break
                    elif state in ["error", "cancelled"]:
                        break
                if not self.autonomous_running or not print_succeeded:
                    break
                
                time.sleep(5)
                self.view.after(0, lambda: self.view.update_status("[Auto] Capturing image..."))
                self.model.capture_and_load_image(UNPROCESSED_IMAGE_FILE)

                final_pil_image = None
                try:
                    user_sigma = self.view.canny_sigma_var.get()
                    debug_mode = self.view.debug_mode_var.get()
                    final_pil_image, _, _ = self.model.run_full_pipeline_with_retry(user_sigma, debug_mode)
                except RuntimeError:
                    self.autonomous_running = False
                    break

                self.view.after(0, lambda: self.view.update_image_display(final_pil_image))
                self.model.save_temp_image(TEMP_IMAGE_FILE)
                self.model.send_telegram_notification(TEMP_IMAGE_FILE, caption="[Auto] Please review.")