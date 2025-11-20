import os

class Config:
    PRINTER_URL = ""
    GCODE_FILE = ""
    TELEGRAM_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    
    CALIBRATION_START_STEP = 0.05
    CALIBRATION_MIN_STEP = 0.01
    MAX_PRINT_RETRIES = 3

    @classmethod
    def load(cls, path='config.txt'):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    lines = [line.strip() for line in f.readlines()]
                    if len(lines) >= 2:
                        cls.PRINTER_URL = lines[0]
                        cls.GCODE_FILE = lines[1]
                    if len(lines) >= 4:
                        cls.TELEGRAM_TOKEN = lines[2]
                        cls.TELEGRAM_CHAT_ID = lines[3]
            except Exception:
                pass

    @classmethod
    def get_webcam_url(cls):
        return f"{cls.PRINTER_URL}/webcam/stream"