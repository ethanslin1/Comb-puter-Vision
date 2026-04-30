import cv2
import numpy as np
from typing import Optional, Tuple


def detect_frame_corners(image: np.ndarray):
    """
    Detect the four outer corners of the hive frame.

    Returns:
        corners: 4 x 2 float32 array, unordered or approximately ordered.
                 Points are image coordinates: (x, y).
        None if detection fails.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Detect edges
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # 4. Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # 5. Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 6. Look for a large 4-corner contour
    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=True)

        approx = cv2.approxPolyDP(
            contour,
            epsilon=0.02 * perimeter,
            closed=True,
        )

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            return corners

    return None


def order_corners(corners: np.ndarray):
    """
    Order corners consistently as:
        top-left, top-right, bottom-right, bottom-left
    """
    corners = np.asarray(corners, dtype=np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    # x + y is smallest at top-left, largest at bottom-right
    s = corners.sum(axis=1)
    ordered[0] = corners[np.argmin(s)]
    ordered[2] = corners[np.argmax(s)]

    # x - y is smallest at bottom-left, largest at top-right
    diff = corners[:, 0] - corners[:, 1]
    ordered[1] = corners[np.argmax(diff)]
    ordered[3] = corners[np.argmin(diff)]

    return ordered


def compute_homography(corners: np.ndarray, output_size: tuple[int, int]):
    """
    Compute homography from detected frame corners to a canonical rectangle.

    Args:
        corners: 4 x 2 corners in order:
                 top-left, top-right, bottom-right, bottom-left
        output_size: (width, height)

    Returns:
        H: 3 x 3 homography matrix
    """
    width, height = output_size

    src = order_corners(corners)

    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src, dst)
    return H

def warp_frame(image: np.ndarray, corners: np.ndarray, output_size: tuple[int, int] = (1024, 1536)):
    """
    Warp the hive frame into a front-facing rectangle.

    Args:
        image: input BGR/RGB image
        corners: detected frame corners
        output_size: (width, height)

    Returns:
        rectified image
    """
    H = compute_homography(corners, output_size)

    rectified = cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR,
    )

    return rectified


def warp_frame(image: np.ndarray, corners: np.ndarray, output_size: tuple[int, int] = (1024, 1536)) -> np.ndarray:
    """
    Warp the hive frame into a front-facing rectangle.

    Args:
        image: input BGR/RGB image
        corners: detected frame corners
        output_size: (width, height)

    Returns:
        rectified image
    """
    H = compute_homography(corners, output_size)

    rectified = cv2.warpPerspective(
        image,
        H,
        output_size,
        flags=cv2.INTER_LINEAR,
    )

    return rectified

def rectify_frame(image: np.ndarray, output_size: tuple[int, int] = (1024, 1536), corners: Optional[np.ndarray] = None):
    """
    Full rectification pipeline.

    If corners are provided, use them.
    Otherwise, try to detect corners automatically.

    Returns:
        rectified: warped frame image
        H: homography matrix
    """
    if corners is None:
        corners = detect_frame_corners(image)

    if corners is None:
        raise ValueError("Could not detect hive frame corners.")

    ordered = order_corners(corners)
    H = compute_homography(ordered, output_size)

    rectified = cv2.warpPerspective(
        image,
        H,
        output_size,
        flags=cv2.INTER_LINEAR,
    )

    return rectified, H

