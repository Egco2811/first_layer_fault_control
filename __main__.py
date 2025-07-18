from capture_image import *
from preprocess_image import *
from send_gcode import *

if __name__ == '__main__':
    printer_address = "http://192.168.1.8"
    capture_image(printer_address, "snapshot.jpg")
    preprocess_image("snapshot.jpg")
    send_gcode(printer_address, "M140 S60")
    send_gcode(printer_address, "M190 S60")
    send_gcode(printer_address, "G28")
    send_gcode(printer_address, "BED_MESH_CALIBRATE")
    send_gcode(printer_address, "G90")
    send_gcode(printer_address, "G0 X150 Y150 F6000")

