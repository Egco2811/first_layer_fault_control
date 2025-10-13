import requests
import json
import time


def capture_image(moonraker_url, save_path):
    """
    Captures an image from the Mainsail webcam.
    """
    try:
        snapshots_url = f"{moonraker_url}/webcam/snapshot"
        response = requests.get(snapshots_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Image captured and saved to {save_path}")
            return save_path
        else:
            print(f"Failed to capture image. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error capturing image: {e}")
        return None


def poll_print_status(moonraker_url, interval=5):
    """
    Polls Moonraker until the printer is no longer in a 'printing' state.
    """
    print("Polling for print status...")
    api_endpoint = f"{moonraker_url}/printer/objects/query?print_stats"
    while True:
        try:
            response = requests.get(api_endpoint)
            response.raise_for_status()
            print_stats = response.json()['result']['status']['print_stats']
            state = print_stats['state']

            progress = 0
            if state == "complete":
                progress = 100.0
            elif 'progress' in print_stats:
                progress = print_stats.get('progress', 0) * 100

            message = f"Print state: '{state}'. Progress: {progress:.1f}%"
            print(message)
            yield state, message

            if state in ["standby", "complete", "error", "cancelled"]:
                if state == "complete":
                    print("Print finished successfully!")
                else:
                    print(f"Print ended with state '{state}'. Message: {print_stats.get('message', 'N/A')}")
                break
            time.sleep(interval)
        except requests.exceptions.RequestException as e:
            error_message = f"Error polling printer status: {e}"
            print(error_message)
            yield "error", error_message
            time.sleep(interval)
        except (KeyError, IndexError) as e:
            error_message = f"Could not parse printer status. Error: {e}"
            print(error_message)
            yield "error", error_message
            time.sleep(interval)


def send_gcode(moonraker_url, command):
    """
    Sends a gcode command to the Moonraker url provided.
    """
    api_endpoint = f"{moonraker_url}/printer/gcode/script"
    parameters = {'script': command}
    try:
        response = requests.post(api_endpoint, params=parameters)
        if response.status_code == 200:
            print(f"Successfully sent G-Code: '{command}'")
            return True
        else:
            print(f"Error sending G-Code: '{command}'")
            print(f"Status Code: {response.status_code}")
            print("Response Body:", response.text)
            return False
    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the HTTP request: {e}")
        return False


def start_print(moonraker_url, gcode_file):
    """
    Sends a command to start a print to the Moonraker url provided.
    """
    api_endpoint = f"{moonraker_url}/printer/print/start"
    payload = {"filename": gcode_file}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        print(f"Successfully sent command to print '{gcode_file}'.")
        print("Response from Moonraker:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def cancel_print(moonraker_url):
    """
    Sends a CANCEL_PRINT command to Moonraker.
    """
    print("Sending CANCEL_PRINT command...")
    return send_gcode(moonraker_url, "CANCEL_PRINT")

def adjust_z_offset(moonraker_url, adjustment):
    """
    Adjusts the Z-offset by the specified amount.
    """
    command = f"SET_GCODE_OFFSET Z_ADJUST={adjustment} MOVE=1"
    print(f"Adjusting Z-offset by {adjustment}...")
    return send_gcode(moonraker_url, command)

def restart_firmware(moonraker_url):
    """
    Sends a FIRMWARE_RESTART command to Moonraker.
    """
    print("Sending FIRMWARE_RESTART command...")
    return send_gcode(moonraker_url, "FIRMWARE_RESTART")

def auto_home(moonraker_url):
    """
    Sends a G28 command to home all axes.
    """
    print("Sending G28 (Auto Home) command...")
    return send_gcode(moonraker_url, "G28")