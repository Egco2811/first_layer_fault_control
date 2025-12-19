import requests
import json
import time

def capture_image(moonraker_url, save_path):
    try:
        url = f"{moonraker_url}/webcam/snapshot"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception:
        return False

def poll_print_status(moonraker_url, interval=5):
    info_url = f"{moonraker_url}/server/info"
    try:
        info_resp = requests.get(info_url, timeout=2)
        if info_resp.status_code == 200:
            klippy_state = info_resp.json().get("result", {}).get("klippy_state")
            if klippy_state in ["error", "shutdown"]:
                yield "error", f"Klipper is in {klippy_state} state"
                return
    except Exception:
        yield "error", "Could not connect to Moonraker"
        return

    url = f"{moonraker_url}/printer/objects/query?print_stats"
    while True:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            stats = response.json()['result']['status']['print_stats']
            state = stats['state']
            
            msg = f"State: {state}"
            if 'progress' in stats:
                msg += f", Progress: {stats['progress']*100:.1f}%"
            
            yield state, msg
            
            if state in ["complete", "error", "cancelled"]:
                break
            
            time.sleep(interval)
        except Exception as e:
            yield "error", str(e)
            time.sleep(interval)

def send_gcode(moonraker_url, command):
    url = f"{moonraker_url}/printer/gcode/script"
    try:
        requests.post(url, json={'script': command}, timeout=5)
        return True  
    except Exception as e:
        print(f"DEBUG: Connection error to {url}: {e}")
        return False

  

def start_print(moonraker_url, filename):
    url = f"{moonraker_url}/printer/print/start"
    try:
        info_url = f"{moonraker_url}/server/info"
        info_resp = requests.get(info_url, timeout=2)
        if info_resp.status_code == 200:
            if info_resp.json().get("result", {}).get("klippy_state") != "ready":
                return False

        requests.post(url, json={"filename": filename}, timeout=5)
        return True
    except Exception:
        return False

def cancel_print(moonraker_url):
    send_gcode(moonraker_url, "CANCEL_PRINT")

def adjust_z_offset(moonraker_url, adjustment):
    send_gcode(moonraker_url, f"SET_GCODE_OFFSET Z_ADJUST={adjustment} MOVE=1")

def apply_and_save_config(moonraker_url):
    send_gcode(moonraker_url, "Z_OFFSET_APPLY_PROBE")
    time.sleep(1)
    send_gcode(moonraker_url, "SAVE_CONFIG")

def restart_firmware(moonraker_url):
    send_gcode(moonraker_url, "FIRMWARE_RESTART")

def auto_home(moonraker_url):
    send_gcode(moonraker_url, "G28")

def wait_for_klipper_ready(moonraker_url, timeout=60):
    url = f"{moonraker_url}/server/info"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                state = response.json().get("result", {}).get("klippy_state")
                if state == "ready":
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def force_stillness(moonraker_url):
    send_gcode(moonraker_url, "M400")
    send_gcode(moonraker_url, "G4 P1500")
    time.sleep(1.5)