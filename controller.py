import threading
import time
import os
import cv2

UNPROCESSED_IMAGE_FILE = "unprocessed.jpg"
TEMP_IMAGE_FILE = "temp_autonomous_capture.jpg"


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.autonomous_running = False

    def _run_task(self, target_func, *args):
        threading.Thread(target=target_func, args=args, daemon=True).start()

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
            self.view.show_error("Processing Error", f"Failed at step '{step_name}':\n\n{e}")
            self.view.update_status(f"Error at step '{step_name}'")

    def process_to_final(self):
        self._run_task(self._process_to_final_task)

    def _process_to_final_task(self):
        self.view.set_ui_state('BUSY')
        self.view.update_status("Running full pipeline...")
        try:
            sigma = self.view.canny_sigma_var.get()
            debug_mode = self.view.debug_mode_var.get()
            final_image = self.model.run_full_pipeline(sigma, debug_mode)
            self.view.update_image_display(final_image)
            self.view.update_status("Full pipeline complete. Ready to save.")
            self.view.set_ui_state('PROCESSED')
        except Exception as e:
            self.view.show_error("Processing Error", f"Failed during full pipeline:\n\n{e}")
            self.view.update_status(f"Error in pipeline: {e}")
            self.view.set_ui_state('CAPTURED')

    def save_final_image(self):
        self._run_task(self._save_image_task)

    def _save_image_task(self):
        self.view.update_status("Saving image...")
        try:
            classification = self.view.classification_var.get()
            save_path = self.model.save_image(classification)
            self.view.show_info("Success", f"Image saved as {save_path}")
            self.view.update_status(f"Saved to '{classification}'. Ready for next cycle.")
        except Exception as e:
            self.view.show_error("Save Error", f"Failed to save image:\n\n{e}")
            self.view.update_status(f"Error saving: {e}")

    def on_sigma_change(self, value):
        self.view.update_status(f"Sigma set to {float(value):.2f}. Press a button to apply.")

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

                self.view.update_status("[Auto] Capturing and processing image...")
                self.model.capture_and_load_image(UNPROCESSED_IMAGE_FILE)
                sigma = self.view.canny_sigma_var.get()
                debug_mode = self.view.debug_mode_var.get()
                final_pil_image = self.model.run_full_pipeline(sigma, debug_mode)
                self.view.update_image_display(final_pil_image)
                if not self.autonomous_running: break

                self.view.update_status("[Auto] Awaiting user acceptance...")
                self.model.save_temp_image(TEMP_IMAGE_FILE)
                self.model.send_telegram_notification(TEMP_IMAGE_FILE,
                                                      caption="[Auto] Please review and accept/reject.")

                if self.view.ask_ok_cancel("Accept Image?",
                                           "The processed image has been sent to Telegram for review.\n\nPress OK to accept and save.\nPress Cancel to reject and continue."):
                    classification = self.view.classification_dropdown.get()
                    save_path = self.model.save_image(classification)
                    self.view.update_status(f"[Auto] Image accepted and saved to {save_path}.")
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