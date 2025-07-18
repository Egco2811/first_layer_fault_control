from capture_image import *
from preprocess_image import *
from send_gcode import *

if __name__ == '__main__':
    printer_address = "http://192.168.1.8"
    capture_image(printer_address, "snapshot.jpg")
    preprocess_image("snapshot.jpg")
    send_gcode(printer_address, "G28")
