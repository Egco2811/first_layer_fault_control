import cv2
import numpy as np

MIN_AREA = 1000
MAX_AREA_RATIO = 0.25
TARGET_SIZE = (256, 256)
PADDING_FACTOR = 1.14

def auto_canny(image, sigma=0.5):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

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
    if not contours: return None

    img_area = processed_image.shape[0] * processed_image.shape[1]
    max_area = img_area * MAX_AREA_RATIO
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > max_area: continue

        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if w == 0 or h == 0: continue
        
        ar = max(w, h) / min(w, h)
        if 0.5 <= ar <= 2.0:
            candidates.append({'ar': ar, 'cnt': c})

    if not candidates: return None

    best = min(candidates, key=lambda x: abs(x['ar'] - 1.0))
    peri = cv2.arcLength(best['cnt'], True)
    approx = cv2.approxPolyDP(best['cnt'], 0.04 * peri, True)

    if len(approx) != 4: return None
    return approx

def crop_from_contour(original_image, contour):
    corners = contour.reshape(4, 2).astype("float32")
    centroid = np.mean(corners, axis=0)
    
    expanded = np.zeros_like(corners)
    for i, c in enumerate(corners):
        expanded[i] = centroid + PADDING_FACTOR * (c - centroid)

    ordered = order_points(expanded)
    dst = np.array([
        [0, 0], [TARGET_SIZE[0]-1, 0],
        [TARGET_SIZE[0]-1, TARGET_SIZE[1]-1],
        [0, TARGET_SIZE[1]-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(original_image, M, TARGET_SIZE)