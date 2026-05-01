import cv2
from geometry.helpers import compute_gradients, harris_cornerness_score 
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import os


def draw_x(image, corners, gray, candidates):
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

    if candidates is not None:
        ax.plot(candidates[:, 0], candidates[:, 1],
                'oy', markersize=8, label="top 8 candidates")
        

    ax.plot(corners[:, 0], corners[:, 1], '+r', markersize=20)  # final 4 in blue
    ax.axis('off')
    ax.set_title("final 4 corners")

    debug_path = "/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/interim/marked_corners/final_four_corners.png"
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    plt.savefig("data/interim/marked_corners/debug_final_corners.png")
    plt.close()

    print(f"Saved debug to {debug_path}")


def validation_check(corners):
    " putpose of this function is a checker for the corner detections to see if they are right"
    "for instance, if the length between the bttom left cornere and top left corner is too different than"
    "the length betwen the top right corner and botto mright corner, then the angle is bad, or the corners"
    "are wrong"
    top_left, top_right, bottom_right, bottom_left = corners

    left_height = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)
    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)

    height_ratio = min(left_height, right_height) / max(left_height, right_height)
    if height_ratio < 0.8:  
        raise ValueError(f"Left/right heights too inconsistent: {left_height:.1f} vs {right_height:.1f} (ratio {height_ratio:.2f})")

    width_ratio = min(top_width, bottom_width) / max(top_width, bottom_width)
    if width_ratio < 0.8:
        raise ValueError(f"Top/bottom widths too inconsistent: {top_width:.1f} vs {bottom_width:.1f} (ratio {width_ratio:.2f})")

    aspect_ratio = max(left_height, right_height) / max(top_width, bottom_width)
    if aspect_ratio < 0.1 or aspect_ratio > 10:
        raise ValueError(f"Aspect ratio looks wrong: {aspect_ratio:.2f}")
    

# def find_best_four(candidates):

#     # take top 8
#     top_n = min(8, len(candidates))
#     candidates = candidates[:top_n]

#     best_corners = None
#     best_score = -1

#     for combo in combinations(range(top_n), 4):
#         pts = candidates[list(combo)]

#         s = pts[:, 0] + pts[:, 1]
#         d = pts[:, 0] - pts[:, 1]
#         ordered = np.array([pts[np.argmin(s)], pts[np.argmax(d)],
#                             pts[np.argmax(s)], pts[np.argmin(d)]])

#         tl, tr, br, bl = ordered

#         left_height  = np.linalg.norm(bl - tl)
#         right_height = np.linalg.norm(br - tr)
#         top_width    = np.linalg.norm(tr - tl)
#         bottom_width = np.linalg.norm(br - bl)

#         if min(left_height, right_height, top_width, bottom_width) < 10:
#             continue

#         height_ratio = min(left_height, right_height) / max(left_height, right_height)
#         width_ratio  = min(top_width, bottom_width)   / max(top_width, bottom_width)

#         score = height_ratio + width_ratio

#         if score > best_score:
#             best_score = score
#             best_corners = ordered

#     if best_corners is None:
#         return None
    
#     return best_corners

def detect_frame_corners(image: np.ndarray):

    "finds corners and takes the top 8  (4 is too strict) to compute rectification"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

    Ix, Iy = compute_gradients(gray)

    h_score = harris_cornerness_score(Ix, Iy)

    from skimage.feature import peak_local_max

    m_dist = 10
    thresh_rel = 0.01

    coords = peak_local_max(h_score, min_distance=m_dist, threshold_rel=thresh_rel)

    candidates = coords[:, ::-1].astype(np.float32) 

    print(f"Number of candidates found: {len(candidates)}") 
    
    if len(candidates) < 4: # most extreme points
        return None
    
    s = candidates[:, 0] + candidates[:, 1]
    d = candidates[:, 0] - candidates[:, 1]

    corners = np.array([candidates[np.argmin(s)],candidates[np.argmax(d)], 
                        candidates[np.argmax(s)],  candidates[np.argmin(d)],   
    ])

    # validation_check(corners)

    # most_rectangualr_corners = find_best_four(candidates)
    
    draw_x(image, corners, gray, candidates)
    

    return corners


def order_corners(found_corners):
    if found_corners is None or len(found_corners) == 0 or len(found_corners) != 4:
        return None

    corners = np.asarray(found_corners, dtype=np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    s = corners.sum(axis=1)
    ordered[0] = corners[np.argmin(s)]
    ordered[2] = corners[np.argmax(s)]

    diff = corners[:, 0] - corners[:, 1]
    ordered[1] = corners[np.argmax(diff)]
    ordered[3] = corners[np.argmin(diff)]

    print("src corners:")
    print(f"  top-left:     {ordered[0]}")
    print(f"  top-right:    {ordered[1]}")
    print(f"  bottom-right: {ordered[2]}")
    print(f"  bottom-left:  {ordered[3]}")

    return ordered


def check_reporojection(src, dst, H):
    print("reprojection check:")
    for i in range(4):
        x, y = src[i]
        pt = H @ np.array([x, y, 1])
        pt = pt / pt[2] 
        print(f"  src {src[i]}, projected {pt[:2]}, expected {dst[i]}")

def compute_homography(corners: np.ndarray, width, height):

    src = order_corners(corners)

    dst = np.array( [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    print("dst corners:")
    print(f"  top-left:     {dst[0]}")
    print(f"  top-right:    {dst[1]}")
    print(f"  bottom-right: {dst[2]}")
    print(f"  bottom-left:  {dst[3]}")

    # H = cv2.getPerspectiveTransform(src, dst)

    A = []
    for i in range(4):
        x, y = src[i]
        xp, yp = dst[i]
        A.append([-x, - y, -1,  0,  0 ,  0,  xp * x,  xp * y,  xp])

        A.append([ 0,  0,  0, -x, -y, -1,  yp * x,  yp * y,  yp])
    
    A = np.array(A)

    _, _, right_singluar_vectors = np.linalg.svd(A)
    
    h = right_singluar_vectors[-1 , :]          
    H = h.reshape(3, 3)  

    H = H  / H[2, 2]

    check_reporojection(src, dst, H)

    return H

def warp(image: np.ndarray, corners: np.ndarray, ):

    width = 5000
    height = 3000

    H = compute_homography(corners, width, height)

    output_size: Tuple[int, int] = (width, height)

    rectified = cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR)

    print(f"After warp - pixel at (0, 0): {rectified[0, 0]}")
    print(f"After warp - pixel at (0, 0): {rectified[0, 0]}")

    return rectified, H

def rectify_frame(image: np.ndarray):

    corners = detect_frame_corners(image)

    print(f"corners: {corners}")

    if corners is None:
        # print("Corners is None")
        raise ValueError("PLEASE RETAKE IMAGE, CORNERS NOT DETECTED")
        # return None

    rectified, H = warp(image, corners)

    print(f"rectified shape: {rectified.shape}")

    return rectified, H

