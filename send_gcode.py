import requests


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
