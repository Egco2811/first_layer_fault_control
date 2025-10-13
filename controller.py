import threading
import time
import os
import cv2
from PIL import Image, ImageTk
from predictor import Predictor
from classifier import train_model


UNPROCESSED_IMAGE_FILE = "unprocessed.jpg"
TEMP_IMAGE_FILE = "temp_autonomous_capture.jpg"


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.predictor = Predictor()
        self.autonomous_running = False
        self.webcam_running = False
        self.start_webcam_stream()

    def _run_task(self, target_func, *args):
        threading.Thread(target=target_func, args=args, daemon=True).start()

    def on_close(self):
        """Handle window closing event."""
        self.stop_webcam_stream()
        self.view.destroy()

    def print_shape(self):
        self._run_task(self._print_shape_task)

    def _print_shape_task(self):
        self.view.set_ui_state('BUSY')
        self.view.update_status("Starting print...")
        try:
            self.model.start_print()
            print_succeeded = False
            for state, message in self.model.poll_print_progress():
                self.view.update_status(message)
                if state == "complete":
                    print_succeeded = True
                    break
                elif state in ["error", "cancelled"]:
                    break

            if print_succeeded:
                self.view.update_status("Print finished.")
            else:
                self.view.update_status("Print failed or was cancelled.")
        except Exception as e:
            self.view.show_error("Print Error", f"Failed to start print:\n\n{e}")
            self.view.update_status(f"Error: {e}")
        finally:
            self.view.set_ui_state('IDLE')

    def capture_and_display_image(self):
        self._run_task(self._capture_task)

    def _capture_task(self):
        self.view.set_ui_state('BUSY')
        self.view.update_status("Capturing image...")
        try:
            image = self.model.capture_and_load_image(UNPROCESSED_IMAGE_FILE)
            self.view.update_image_display(image)
            self.view.update_status("Image captured. Ready to process.")
            self.view.set_ui_state('CAPTURED')
        except Exception as e:
            self.view.show_error("Capture Error", f"Failed to capture image:\n\n{e}")
            self.view.update_status(f"Error: {e}")
            self.view.set_ui_state('IDLE')

    def view_step(self, step_name):
        self._run_task(self._view_step_task, step_name)

    def _view_step_task(self, step_name):
        self.view.update_status(f"Applying '{step_name}'...")
        try:
            sigma = self.view.canny_sigma_var.get()
            debug_mode = self.view.debug_mode_var.get()
            processed_image = self.model.process_image_step(step_name, sigma, debug_mode)
            self.view.update_image_display(processed_image)

            if step_name == 'Crop to Shape':
                self.view.update_status(f"Applied '{step_name}'. Ready to save.")
                self.view.set_ui_state('PROCESSED')
            else:
                self.view.update_status(f"Applied '{step_name}' filter.")
        except Exception as e:
            if step_name == 'Crop to Shape' and isinstance(e, RuntimeError):
                self.view.update_status("Shape detection failed. Trying fallback...")
                try:
                    fallback_image = self.model.crop_with_fallback()
                    self.view.update_image_display(fallback_image)
                    self.view.update_status("Used last successful crop parameters. Ready to save.")
                    self.view.set_ui_state('PROCESSED')
                    self.view.show_info("Processing Notice",
                                        "Could not detect a new shape. The last successful crop parameters have been applied.")
                except Exception as fallback_e:
                    self.view.show_error("Processing Error",
                                         f"Failed at step '{step_name}':\n\n{e}\n\nFallback also failed:\n\n{fallback_e}")
                    self.view.update_status(f"Error at step '{step_name}'")
            else:
                self.view.show_error("Processing Error", f"Failed at step '{step_name}':\n\n{e}")
                self.view.update_status(f"Error at step '{step_name}'")

    def process_to_final(self):
        self._run_task(self._process_to_final_task)

    def _process_to_final_task(self):
        self.view.set_ui_state('BUSY')
        self.view.update_status("Running full pipeline...")
        try:
            user_sigma = self.view.canny_sigma_var.get()
            debug_mode = self.view.debug_mode_var.get()
            final_image, successful_sigma, used_fallback = self.model.run_full_pipeline_with_retry(user_sigma,
                                                                                                   debug_mode)

            self.view.update_image_display(final_image)
            status_message = f"Pipeline succeeded with sigma={successful_sigma:.2f}. Ready to save."
            if used_fallback:
                status_message = "Shape detection failed. Used last successful parameters. Ready to save."
                self.view.show_info("Processing Notice",
                                    "Could not detect a new shape. The last successful crop parameters have been applied.")

            self.view.update_status(status_message)
            self.view.set_ui_state('PROCESSED')

        except RuntimeError as e:
            self.view.show_error("Processing Error", str(e))
            self.view.update_status("Error in pipeline: Could not find a suitable shape.")
            self.view.set_ui_state('CAPTURED')

        except Exception as e:
            self.view.show_error("Processing Error", f"An unexpected error occurred:\n\n{e}")
            self.view.update_status(f"Error in pipeline: {e}")
            self.view.set_ui_state('CAPTURED')

    def save_final_image(self):
        self._run_task(self._save_image_task)

    def _save_image_task(self):
        self.view.update_status("Saving image...")
        try:
            classification = self.view.classification_var.get()
            save_path = self.model.save_image(classification)
            self.model.commit_last_found_corners()
            self.view.show_info("Success", f"Image saved as {save_path}")
            self.view.update_status(f"Saved to '{classification}'. Ready for next cycle.")
        except Exception as e:
            self.view.show_error("Save Error", f"Failed to save image:\n\n{e}")
            self.view.update_status(f"Error saving: {e}")

    def on_sigma_change(self, value):
        self.model.invalidate_sigma_dependent_cache()
        self.view.update_status(f"Sigma set to {float(value):.2f}. Cache cleared. Press a button to apply.")

    def toggle_autonomous_mode(self):
        if not self.autonomous_running:
            self.autonomous_running = True
            self.view.set_ui_state('AUTONOMOUS')
            self._run_task(self.autonomous_loop)
        else:
            self.autonomous_running = False
            self.view.update_status("Stopping autonomous mode and cancelling print...")
            self._run_task(self.model.cancel_print)

    def autonomous_loop(self):
        while self.autonomous_running:
            try:
                self.view.update_status("[Auto] Starting print...")
                self.model.start_print()
                print_succeeded = False
                for state, message in self.model.poll_print_progress():
                    if not self.autonomous_running:
                        break
                    self.view.update_status(f"[Auto] {message}")
                    if state == "complete":
                        print_succeeded = True
                        break
                    elif state in ["error", "cancelled"]:
                        self.view.update_status(f"[Auto] Print failed or was cancelled. State: {state}")
                        break

                if not self.autonomous_running or not print_succeeded:
                    if print_succeeded:
                        self.view.update_status("[Auto] Operation cancelled by user.")
                    else:
                        self.view.update_status("[Auto] Print did not complete. Stopping.")
                    break
                time.sleep(5)

                self.view.update_status("[Auto] Capturing image...")
                self.model.capture_and_load_image(UNPROCESSED_IMAGE_FILE)

                self.view.update_status("[Auto] Processing with retry logic...")
                final_pil_image = None
                try:
                    user_sigma = self.view.canny_sigma_var.get()
                    debug_mode = self.view.debug_mode_var.get()
                    final_pil_image, successful_sigma, used_fallback = self.model.run_full_pipeline_with_retry(
                        user_sigma, debug_mode)
                    status_message = f"[Auto] Image processed successfully with sigma={successful_sigma:.2f}."
                    if used_fallback:
                        status_message = "[Auto] Shape detection failed. Used last successful parameters."
                    self.view.update_status(status_message)

                except RuntimeError as e:
                    error_msg = f"[Auto] {e}. Stopping autonomous mode."
                    self.view.update_status(error_msg)
                    self.view.show_error("Autonomous Error", error_msg)
                    self.autonomous_running = False

                if not self.autonomous_running: break

                self.view.update_image_display(final_pil_image)
                if not self.autonomous_running: break

                self.view.update_status("[Auto] Awaiting user acceptance...")
                self.model.save_temp_image(TEMP_IMAGE_FILE)
                self.model.send_telegram_notification(TEMP_IMAGE_FILE,
                                                      caption="[Auto] Please review and accept/reject.")

                acceptance = self.view.ask_accept_fallback_reject("Review Image",
                                                                  "The processed image has been sent for review.\n\n"
                                                                  "YES: Accept and save this image.\n"
                                                                  "NO: Reject this and save using the last good coordinates (fallback).\n"
                                                                  "CANCEL: Reject this image and discard.")

                if acceptance is True:
                    classification = self.view.classification_var.get()
                    save_path = self.model.save_image(classification)
                    self.model.commit_last_found_corners()
                    self.view.update_status(f"[Auto] Image accepted and saved to {save_path}.")
                elif acceptance is False:
                    self.view.update_status("[Auto] User rejected current image. Applying fallback...")
                    try:
                        fallback_pil_image = self.model.crop_with_fallback()
                        self.view.update_image_display(fallback_pil_image)
                        classification = self.view.classification_var.get()
                        save_path = self.model.save_image(classification)
                        self.view.update_status(f"[Auto] Fallback image saved to {save_path}.")
                    except Exception as e:
                        error_msg = f"[Auto] Fallback processing failed: {e}. Discarding image."
                        self.view.show_error("Autonomous Error", error_msg)
                        self.view.update_status(error_msg)
                else:  
                    self.view.update_status("[Auto] Image rejected by user. Discarding.")

                if os.path.exists(TEMP_IMAGE_FILE):
                    os.remove(TEMP_IMAGE_FILE)

                if not self.autonomous_running: break

                self.view.update_status("[Auto] Clean build plate and press OK to continue.")
                if not self.view.ask_ok_cancel("Autonomous Mode",
                                               "Clean build plate and press OK to continue.\n\nPress Cancel to stop."):
                    self.autonomous_running = False

            except Exception as e:
                self.view.show_error("Autonomous Error", f"A critical error occurred:\n\n{e}")
                self.autonomous_running = False

        self.autonomous_running = False
        self.view.set_ui_state('IDLE')
        self.view.update_status("Idle")

    def adjust_z(self, amount):
        self._run_task(self._adjust_z_task, amount)

    def _adjust_z_task(self, amount):
        self.view.update_status(f"Adjusting Z-Offset by {amount:.3f}mm...")
        try:
            self.model.adjust_z_offset(amount)
            self.view.update_status(f"Z-Offset adjusted. Ready.")
        except Exception as e:
            self.view.show_error("Z-Offset Error", f"Failed to adjust Z-Offset:\n\n{e}")
            self.view.update_status(f"Error: {e}")

    def restart_firmware(self):
        if self.view.ask_ok_cancel("Confirm Restart", "Are you sure you want to restart the printer firmware?"):
            self._run_task(self._restart_firmware_task)

    def _restart_firmware_task(self):
        self.view.set_ui_state('BUSY')
        self.view.update_status("Restarting firmware...")
        try:
            self.model.restart_firmware()
            self.view.update_status("Firmware restart command sent.")
            self.view.show_info("Firmware Restart", "Firmware restart command sent. The printer will reboot.")
        except Exception as e:
            self.view.show_error("Firmware Restart Error", f"Failed to send restart command:\n\n{e}")
            self.view.update_status(f"Error: {e}")
        finally:
            self.view.set_ui_state('IDLE')

    def auto_home(self):
        self._run_task(self._auto_home_task)

    def _auto_home_task(self):
        self.view.set_ui_state('BUSY')
        self.view.update_status("Homing all axes...")
        try:
            self.model.auto_home()
            self.view.update_status("Homing complete.")
        except Exception as e:
            self.view.show_error("Homing Error", f"Failed to home axes:\n\n{e}")
            self.view.update_status(f"Error: {e}")
        finally:
            self.view.set_ui_state('IDLE')

    def send_gcode(self, command):
        if not command:
            return
        self._run_task(self._send_gcode_task, command)

    def _send_gcode_task(self, command):
        self.view.update_status(f"Sending G-Code: {command}")
        try:
            self.model.send_gcode(command)
            self.view.update_status(f"G-Code '{command}' sent successfully.")
        except Exception as e:
            self.view.show_error("G-Code Error", f"Failed to send command:\n\n{e}")
            self.view.update_status(f"Error sending G-Code: {e}")

    def start_webcam_stream(self):
        if not self.webcam_running:
            self.webcam_running = True
            self._run_task(self._webcam_loop)

    def stop_webcam_stream(self):
        self.webcam_running = False

    def _webcam_loop(self):
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '0'

        stream_url = self.model.get_webcam_stream_url()
        self.view.update_status(f"Starting webcam stream from {stream_url}")

        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

        while self.webcam_running:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_image.thumbnail((240, 180))  
                self.view.update_webcam_display(pil_image)
            else:
                self.view.update_status("Webcam stream ended or failed to connect. Retrying...")
                time.sleep(5) 
                cap.release()
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

        cap.release()
        self.view.update_status("Webcam stream stopped.")

    def set_prediction_file(self, file_path):
        self.prediction_file_path = file_path
        self.view.update_classifier_console(f"Selected file for prediction: {file_path}")

    def predict_selected_file(self):
        if not hasattr(self, "prediction_file_path") or not self.prediction_file_path:
            self.view.update_classifier_console("No file selected for prediction.")
            return
        threading.Thread(target=self._predict_task, args=(self.prediction_file_path,), daemon=True).start()

    def _predict_task(self, image_path):
        self.view.update_classifier_console("Running prediction...")
        try:
            predicted_class, confidence = self.predictor.predict(image_path)
            result = f"Prediction: '{predicted_class}' ({confidence:.2f}% confidence)"
            self.view.update_classifier_console(result)
        except Exception as e:
            self.view.update_classifier_console(f"Prediction failed: {e}")

    def start_network_training(self):
        self.view.update_classifier_console("Network training started...")
        try:
            history = train_model()
            final_acc = history.history.get('accuracy', [None])[-1]
            self.view.update_classifier_console(f"Training completed. Final accuracy: {final_acc:.2f}" if final_acc else "Training completed successfully.")
        except Exception as e:
            self.view.update_classifier_console(f"Training failed: {e}")