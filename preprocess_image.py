import cv2
import numpy as np


def auto_canny(image, sigma=0.5):
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


def find_target_contour(processed_image, debug_mode=False):
    contours, _ = cv2.findContours(processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in the processed image.")

    candidates = []

    image_height, image_width = processed_image.shape[:2]
    total_image_area = image_height * image_width
    max_contour_area = total_image_area * 0.25

    debug_image = None
    if debug_mode:
        debug_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    for c in contours:
        area = cv2.contourArea(c)

        if debug_mode:
            cv2.drawContours(debug_image, [c], -1, (255, 0, 0), 2)

        if area < 1000 or area > max_contour_area:
            continue

        rect = cv2.minAreaRect(c)
        (x, y), (width, height), angle = rect

        if width == 0 or height == 0:
            continue

        box = cv2.boxPoints(rect)
        box = np.int_(box)

        if debug_mode:
            cv2.drawContours(debug_image, [box], -1, (0, 0, 255), 2)

        aspect_ratio = max(width, height) / min(width, height)

        if debug_mode:
            text_pos = (int(x - width / 2), int(y))
            cv2.putText(debug_image, f"A:{area:.0f}", (text_pos[0], text_pos[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            cv2.putText(debug_image, f"R:{aspect_ratio:.2f}", (text_pos[0], text_pos[1] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        if 0.5 <= aspect_ratio <= 2.0:
            candidates.append({'aspect_ratio': aspect_ratio, 'raw_contour': c, 'area': area})

    if not candidates:
        if debug_mode:
            cv2.imwrite("debug_shape_finding_failure.jpg", debug_image)
            raise RuntimeError("Could not find any suitable shape. A debug image has been saved.")
        else:
            raise RuntimeError("Could not find any suitable shape. Enable Debug Mode for a visual report.")

    best_candidate = min(candidates, key=lambda x: abs(x['aspect_ratio'] - 1.0))

    perimeter = cv2.arcLength(best_candidate['raw_contour'], True)
    approx_corners = cv2.approxPolyDP(best_candidate['raw_contour'], 0.04 * perimeter, True)

    if len(approx_corners) != 4:
        raise RuntimeError(
            f"Best candidate was not a 4-sided shape after approximation. Found {len(approx_corners)} corners.")

    return approx_corners


def crop_from_contour(original_image, contour):
    corners = contour.reshape(4, 2).astype("float32")

    centroid = np.mean(corners, axis=0)

    padding_scale_factor = 1.14

    expanded_corners = np.zeros_like(corners)
    for i, corner in enumerate(corners):
        vector = corner - centroid
        expanded_corners[i] = centroid + padding_scale_factor * vector

    ordered_expanded_corners = order_points(expanded_corners)

    output_size = (256, 256)
    dst_points = np.array([
        [0, 0], [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(ordered_expanded_corners, dst_points)
    warped_image = cv2.warpPerspective(original_image, transform_matrix, output_size)
    return warped_image