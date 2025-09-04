from preprocess_image import *
from moonraker_interface import *
from file_handler import *

with open('credentials.txt', 'r') as cred_file:
    token = cred_file.readline().strip()
    chat_id = cred_file.readline().strip()

if __name__ == '__main__':
    classification = "ideal"
    printer_address = "http://192.168.1.8"
    counter = count_images(f"images/{classification}/")
    while True:
        try:
            processed_image_file = "snapshot.jpg"
            unprocessed_image_file = "unprocessed.jpg"
            start_print(printer_address, "calibration_shape.gcode")
            check_print_finish(printer_address)
            time.sleep(5)
            capture_image(printer_address, unprocessed_image_file)
            try:
                preprocess_image(unprocessed_image_file)
                send_telegram_image(token, chat_id, processed_image_file)
                accepted = input("Do you wish to continue? (y/n) ")
                if accepted.lower() != 'y':
                    os.makedirs(f"images/{classification}/", exist_ok=True)
                    os.rename(processed_image_file, f"images/{classification}/{counter}.jpg")
                    counter += 1
            except Exception as e:
                print(e)
                send_telegram_image(token, chat_id, unprocessed_image_file, caption="Could not find calibration shape!")
            input("Clean the build plate and press enter to continue...")
        except Exception as e:
            print(f"An error occurred: {e}")
            break


