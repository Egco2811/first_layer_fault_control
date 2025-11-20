import os
import json

class Config:
    PRINTER_URL = "http://192.168.0.8"
    GCODE_FILE = "calibration_shape.gcode"
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    
    @classmethod
    def load(cls, path='config.txt'):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    lines = [line.strip() for line in f.readlines()]
                    if len(lines) >= 4:
                        cls.PRINTER_URL = lines[0]
                        cls.GCODE_FILE = lines[1]
                        cls.TELEGRAM_TOKEN = lines[2]
                        cls.TELEGRAM_CHAT_ID = lines[3]
            except Exception:
                pass

    @classmethod
    def get_webcam_url(cls):
        return f"{cls.PRINTER_URL}/webcam/stream"

    @classmethod
    def get_snapshot_url(cls):
        return f"{cls.PRINTER_URL}/webcam/snapshot"