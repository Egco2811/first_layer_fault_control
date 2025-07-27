from preprocess_image import *
from moonraker_interface import *
from file_handler import *

if __name__ == '__main__':
    printer_address = "http://192.168.1.8"
    high_counter = count_images("images/high/")
    ideal_counter = count_images("images/ideal/")
    low_counter = count_images("images/low/")
    while True:
        try:
            image_file = "unprocessed.jpg"
            start_print(printer_address, "calibration_shape.gcode")
            check_print_finish(printer_address)
            send_gcode(printer_address, "G0 X220 Y90 F6000")
            capture_image(printer_address, image_file)
            preprocess_image(image_file)
            classification = input("Classify the image (high/ideal/low): ").strip().lower()
            if classification == "high":
                os.rename(image_file, f"images/high/{high_counter}.jpg")
                high_counter += 1
            elif classification == "ideal":
                os.rename(image_file, f"images/ideal/{ideal_counter}.jpg")
                ideal_counter += 1
            elif classification == "low":
                os.rename(image_file, f"images/low/{low_counter}.jpg")
                low_counter += 1
            else:
                print("Invalid classification. Please enter 'high', 'ideal', or 'low'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            break


