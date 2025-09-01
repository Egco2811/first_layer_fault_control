import cv2
import numpy as np


def preprocess_image(image_path):
    """
    Preprocesses the image by detecting edges of the calibration shape
    and cropping the image to contain only the calibration shape.

    Args:
        image_path (str): Path of the image file.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise "Error: Could not read the image."
    cropped_image = image[200:400, 280:480]
    grayed_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayed_image, (5, 5), 1)
    edges = auto_canny(blurred_image, 0.3)
    kernel = np.ones((7, 7), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    longest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(longest_contour, True)
    epsilon = 0.04 * perimeter
    approximate_polygon = cv2.approxPolyDP(longest_contour, epsilon, True)
    if len(approximate_polygon) == 4:
        print("Found calibration shape")
        offset_corners = approximate_polygon.reshape(4, 2) + np.array([280, 200])

        ordered_corners = order_points(offset_corners.astype("float32"))

        output_size = (256, 256)
        padding = 20

        dest_side_length = output_size[0] - (padding * 2)

        dst_points = np.array([
            [padding, padding],
            [padding + dest_side_length - 1, padding],
            [padding + dest_side_length - 1, padding + dest_side_length - 1],
            [padding, padding + dest_side_length - 1]
        ], dtype="float32")

        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)

        final_image = cv2.warpPerspective(image, transform_matrix, output_size)
        cv2.imwrite("snapshot.jpg", final_image)
    else:
        raise "Could not find calibration shape"
    return True


def auto_canny(image, sigma=0.5):
    """
    Automatically calculates edges of the calibration shape

    Args:
        image(str): Name of cv2 image.
        sigma(float): Sigma parameter for edge detection.
    """
    intensities = float(np.median(image))

    lower_bound = int(max(0.0, (1.0 - sigma) * intensities))
    upper_bound = int(min(255.0, (1.0 + sigma) * intensities))
    edges = cv2.Canny(image, lower_bound, upper_bound)
    return edges

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
