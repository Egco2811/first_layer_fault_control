import requests


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
