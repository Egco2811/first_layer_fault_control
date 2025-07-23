from preprocess_image import *
from moonraker_interface import *

if __name__ == '__main__':
    printer_address = "http://192.168.1.8"
    start_print(printer_address, "calibration_shape.gcode")
    send_gcode(printer_address, "G0 X150 Y150 F6000")
    capture_image(printer_address, "snapshot.jpg")
    preprocess_image("snapshot.jpg")

    
