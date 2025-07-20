import cv2

def preprocess_image(image_path):
    """
    Preprocesses the image by detecting edges of the calibration shape
    and cropping the image to contain only the calibration shape.

    Args:
        image_path (str): Path of the image file.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read the image.")
        return False

    cropped_image = image[205:395, 285:475]
    grayed_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayed_image, (5, 5), 3)
    edges = cv2.Canny(image=blurred_image, threshold1=30, threshold2=100)

    return True
