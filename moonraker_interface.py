import requests
import json


def capture_image(moonraker_url, save_path):
    """
    Captures an image from the Mainsail webcam.

    Args:
        moonraker_url (str): The URL of the Moonraker server.
        save_path (str): The path to save the captured image.
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


def send_gcode(moonraker_url, command):
    """
    Sends a gcode command to the Moonraker url provided.

    Args:
        moonraker_url (str): The URL of the Moonraker server.
        command (str): The GCode command to be sent.
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

    Args:
        moonraker_url (str): The URL of the Moonraker server.
        gcode_file (str): The GCode file to be printed.
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
