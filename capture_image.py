import requests

def capture_image(snapshot_url, save_path):
    """
    Captures an image from the Mainsail webcam.

    Args:
        snapshot_url (str): The URL of the webcam snapshot.
        save_path (str): The path to save the captured image.
    """
    try:
        response = requests.get(snapshot_url)
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

