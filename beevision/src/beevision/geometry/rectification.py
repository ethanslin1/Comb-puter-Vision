import cv2
from geometry.helpers import compute_gradients, harris_cornerness_score 
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def draw_x(image, corners, gray):
    # debug = image.copy()
    # for (x, y) in corners:
    #     x, y = int(x), int(y)
    #     size = 20
    #     thickness = 3
    #     color = (0, 0, 255)  # red in BGR
    #     cv2.line(debug, (x - size, y - size), (x + size, y + size), color, thickness)
    #     cv2.line(debug, (x + size, y - size), (x - size, y + size), color, thickness)

    # import os
    # debug_path = "data/interim/marked_corners/debug_corners.jpg"
    # os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    # cv2.imwrite(debug_path, debug)
    # print(f"Saved debug corners to {debug_path}")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(gray, cmap=plt.cm.gray)
    ax.plot(corners[:, 0], corners[:, 1], '+b', markersize=20)  # final 4 in blue
    ax.axis('off')
    ax.set_title("Final 4 corners")
    plt.savefig("data/interim/marked_corners/debug_final_corners.png")
    plt.close()

def detect_frame_corners(image: np.ndarray):
    """
    Detect the four outer corners of the hive frame.

    Returns:
        corners: 4 x 2 float32 array, unordered or approximately ordered. The points are image coordinates: (x, y).
        Returns none if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

    Ix, Iy = compute_gradients(gray)

    h_score = harris_cornerness_score(Ix, Iy)

    from skimage.feature import peak_local_max
    coords = peak_local_max(h_score, min_distance=50, threshold_rel=0.01)

    candidates = coords[:, ::-1].astype(np.float32)  
    
    if len(candidates) < 4: # most extreme points
        return None
    
    s = candidates[:, 0] + candidates[:, 1]
    d = candidates[:, 0] - candidates[:, 1]

    corners = np.array([candidates[np.argmin(s)],candidates[np.argmax(d)], 
                        candidates[np.argmax(s)],  candidates[np.argmin(d)],   
    ])

    draw_x(image, corners, gray)

    print("Detected corners:")
    print(f"  top-left:     {corners[0]}")
    print(f"  top-right:    {corners[1]}")
    print(f"  bottom-right: {corners[2]}")
    print(f"  bottom-left:  {corners[3]}")

    return corners


def order_corners(found_corners):
    """
    order corners to top-left, top-right, bottom-right, bottom-left
    """
    if found_corners is None or len(found_corners) == 0 or len(found_corners) != 4:
        return None

    corners = np.asarray(found_corners, dtype=np.float32)

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
    compute homogrpahy given the four courners to 'flatten' the image

    arguments:
        corners: 4 x 2 corners in order: top-left, top-right, bottom-right, bottom-left
        output_size: (width, height)

    returns: H: 3 x 3 homography matrix
    """
    width, height = output_size

    src = order_corners(corners)

    dst = np.array( [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

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

    rectified = cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR,)

    return rectified

def rectify_frame(image: np.ndarray, output_size: tuple[int, int] = (1024, 1536)):
    """
    Full rectification pipeline.

    If corners are provided, use them.
    Otherwise, try to detect corners automatically.

    Returns:
        rectified: warped frame image
        H: homography matrix
    """
    corners = detect_frame_corners(image)

    ordered = order_corners(corners)
    H = compute_homography(ordered, output_size)

    rectified = cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR)

    return rectified, H

