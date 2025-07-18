import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocesses the image by removing the background
    and saves it to the given path.

    Args:
        image_path (str): Path of the image file.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return False
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, image, mask=mask)
    x_start = max(x - 20, 0)
    y_start = max(y - 20, 0)
    x_end = min(x + w + 20, image.shape[1])
    y_end = min(y + h + 20, image.shape[0])

    cropped_image = result[y_start:y_end, x_start:x_end]
    cv2.imwrite(image_path, cropped_image)
    print(f"Processed image saved to {image_path}")
    return True