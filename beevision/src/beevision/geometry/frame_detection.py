import cv2
import numpy as np

def detect_frame_corners(image: np.ndarray) -> np.ndarray:
    """
    Returns 4 corners (x, y) in consistent order:
    [top-left, top-right, bottom-right, bottom-left]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray = np.float32(gray) 
    corners = cv2.goodFeaturesToTrack(  # detect corners (Shi-Tomasi is more stable than Harris for this task)
        gray,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=20
    )

    if corners is None:
        return None

    corners = corners.reshape(-1, 2)

    x_sorted = corners[np.argsort(corners[:, 0])]    # approximate frame equals bounding box of strongest structure
    left = x_sorted[:10]
    right = x_sorted[-10:]

    top_left = left[np.argmin(left[:, 1])]
    bottom_left = left[np.argmax(left[:, 1])]
    top_right = right[np.argmin(right[:, 1])]
    bottom_right = right[np.argmax(right[:, 1])]

    frame = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    return frame