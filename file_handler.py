import os
import requests

def count_images(directory):
    count = 0
    for root, _, files in os.walk(directory):
        count += len([f for f in files if f.endswith('.jpg')])
    return count

def send_telegram_image(bot_token, chat_id, image_path, caption="Print finished!"):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(image_path, "rb") as image_file:
        files = {"photo": image_file}
        data = {"chat_id": chat_id, "caption": caption}
        response = requests.post(url, data=data, files=files)
    return response.ok