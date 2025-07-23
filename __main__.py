from preprocess_image import *
from moonraker_interface import *

if __name__ == '__main__':
    printer_address = "http://192.168.1.8"
    start_print(printer_address, "calibration_shape.gcode")
    check_print_finish(printer_address)
    send_gcode(printer_address, "G0 X220 Y90 F6000")
    capture_image(printer_address, "snapshot.jpg")
    preprocess_image("snapshot.jpg")


