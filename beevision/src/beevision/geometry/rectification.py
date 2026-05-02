import cv2
from geometry.helpers import compute_gradients, harris_cornerness_score, warp_helper, rectangle_score, fit_line_regression, point_line_distance, filter_corners_and_edges_close_to_approximate_frame_edge, show_edges
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import os


def draw_x(image, corners, gray, candidates, valid_quads, quadrants, index, approx_lines):
    "marks all found corners iwth a green circle, the chosen red corners wth a red x, and illustrates if quad was abel to find a corner "
    "or not for explainability/visibility"

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
        

    ax.plot(corners[:, 0], corners[:, 1], '+r', markersize = 20)  # final 4 in blue
    ax.axis('off')
    ax.set_title("final 4 corners")


    for i, ((xmin, xmax, ymin, ymax), is_valid) in enumerate(zip(quadrants, valid_quads)): 

        if is_valid:
            color = "green"
        else:
            color = "red"

        rect_x = [xmin, xmax, xmax, xmin, xmin]
        rect_y = [ymin, ymin, ymax, ymax, ymin]

        ax.plot(rect_x, rect_y, color = color, linewidth = 2)

    # draw the lines for visibiltiy of linear regression from approx_lines
    if approx_lines is not None:
        x_vals = np.linspace(0, approx_lines['width'], 200)
        y_vals = np.linspace(0, approx_lines['height'], 200)

        for key, color in [('top', 'blue'), ('bottom', 'cyan')]:
            line = approx_lines.get(key)
            if line is not None:
                a, b, c = line
                y_line = (-a * x_vals - c) / b
                ax.plot(x_vals, y_line, color=color, linewidth=1.5,
                        linestyle='--', label=f'{key} line')

        for key, color in [('left', 'magenta'), ('right', 'orange')]:
            line = approx_lines.get(key)
            if line is not None:
                a, b, c = line
                x_line = (-b * y_vals - c) / a
                ax.plot(x_line, y_vals, color=color, linewidth=1.5,
                        linestyle='--', label=f'{key} line')

    ax.legend()

    debug_path = f"/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/src/beevision/data/interim/marked_corners/final_four_corners_{index}.png"
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    plt.savefig(debug_path)
    plt.close()

    print(f"Saved debug to {debug_path}")


# def validation_check(corners):
#     " putpose of this function is a checker for the corner detections to see if they are right"
#     "for instance, if the length between the bttom left cornere and top left corner is too different than"
#     "the length betwen the top right corner and botto mright corner, then the angle is bad, or the corners"
#     "are wrong"
#     top_left, top_right, bottom_right, bottom_left = corners

#     left_height = np.linalg.norm(bottom_left - top_left)
#     right_height = np.linalg.norm(bottom_right - top_right)
#     top_width = np.linalg.norm(top_right - top_left)
#     bottom_width = np.linalg.norm(bottom_right - bottom_left)

#     height_ratio = min(left_height, right_height) / max(left_height, right_height)
#     if height_ratio < 0.8:  
#         raise ValueError(f"Left/right heights too inconsistent: {left_height:.1f} vs {right_height:.1f} (ratio {height_ratio:.2f})")

#     width_ratio = min(top_width, bottom_width) / max(top_width, bottom_width)
#     if width_ratio < 0.8:
#         raise ValueError(f"Top/bottom widths too inconsistent: {top_width:.1f} vs {bottom_width:.1f} (ratio {width_ratio:.2f})")

#     aspect_ratio = max(left_height, right_height) / max(top_width, bottom_width)
#     if aspect_ratio < 0.1 or aspect_ratio > 10:
#         raise ValueError(f"Aspect ratio looks wrong: {aspect_ratio:.2f}")

def validation_check(corners, image, index):
    "checks if detected corners form a consistent rectangle"

    
    if len(corners) < 4:
        return False

    # order corners first so we know exactly which is which
    ordered = order_corners(corners)
    if ordered is None:
        return False

    tl = ordered[0]  # top-left
    tr = ordered[1]  # top-right
    br = ordered[2]  # bottom-right
    bl = ordered[3]  # bottom-left

    # left side:  top-left → bottom-left
    left_height  = np.linalg.norm(bl - tl)
    # right side: top-right → bottom-right
    right_height = np.linalg.norm(br - tr)
    # top side:   top-left → top-right
    top_width    = np.linalg.norm(tr - tl)
    # bottom side: bottom-left → bottom-right
    bottom_width = np.linalg.norm(br - bl)

    print(f"Left height:   {left_height:.1f}")
    print(f"Right height:  {right_height:.1f}")
    print(f"Top width:     {top_width:.1f}")
    print(f"Bottom width:  {bottom_width:.1f}")

    image_h, image_w = image.shape[:2]
    image_diagonal = np.sqrt(image_w**2 + image_h**2)
    height_threshold = image_diagonal * 0.10
    width_threshold  = image_diagonal * 0.10

    error = False

    # check 1: |left_height - right_height| > threshold
    height_diff = abs(left_height - right_height)
    print(f"Height difference: {height_diff:.1f} (threshold: {height_threshold})")
    if height_diff > height_threshold:
        error = True

    # check 2: |top_width - bottom_width| > threshold
    width_diff = abs(top_width - bottom_width)
    print(f"Width difference:  {width_diff:.1f} (threshold: {width_threshold})")
    if width_diff > width_threshold:
        error = True

    if error == False:
        print("Validation passed!")
        return True

    # draw warning on image and save
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ax.imshow(gray, cmap='gray')

    ordered = np.array([tl, tr, br, bl])
    ax.plot(ordered[:, 0], ordered[:, 1], '+r', markersize=20)

    # draw sides colored by pass/fail
    sides = [ (tl, tr, f"top: {top_width:.0f}px",       width_diff  <= width_threshold), 
             (bl, br, f"bottom: {bottom_width:.0f}px",  width_diff  <= width_threshold),
             (tl, bl, f"left: {left_height:.0f}px",     height_diff <= height_threshold),
             (tr, br, f"right: {right_height:.0f}px",   height_diff <= height_threshold),
    ]

    for p1, p2, label, passed in sides:
        color = 'green' if passed else 'red'
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=3)
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        ax.text(mid_x, mid_y, label, color=color, fontsize=10,
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    warning_text = "RETAKE PHOTO AT BETTER ANGLE\n"
    ax.text(0.5, 0.05, warning_text, transform=ax.transAxes,
            fontsize=12, color='red', ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_title("VALIDATION FAILED - RETAKE PHOTO", color='red', fontsize=14)
    ax.axis('off')

    debug_path = f"/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/src/beevision/data/interim/marked_corners/validation_fail_{index}.png"
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    plt.savefig(debug_path)
    plt.close()

    print(f"VALIDATION FAILED:")
    # for issue in issues:
    #     print(f"  → {issue}")
    print(f"Saved validation fail image to {debug_path}")

    return False

    

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

def is_corner(Ix, Iy, x, y, window=3, ratio_thresh=0.05):
    Ix2 = Ix[y - window: y + window +1, x-window: x + window +1] **2
    Iy2 = Iy[y - window: y + window +1, x -window: x +window+ 1] **2
    Ixy = Ix[y - window: y + window +1, x-window: x +window +1] * Iy[y-window: y + window + 1, x-window: x + window + 1]

    Sxx = Ix2.sum()
    Syy = Iy2.sum()
    Sxy = Ixy.sum()

    M = np.array([[Sxx, Sxy], [Sxy, Syy]])

    # eigvals = np.linalg.eigvals(M)
    # eigvals = np.sort(eigvals)

    eigvals = np.linalg.eigvalsh(M) 

    l1, l2 = eigvals[1], eigvals[0]

    ratio = l2 / (l1 + 1e-6)

    # ratio = eigvals[0] / (eigvals[1] + 1e-6) # lecture isnpired: if eiegnvaleu dominate,s then it is an edge
    return ratio > ratio_thresh




# def refine_rectangle_ransac(candidates, init_corners, quadrants,
#                             score_thresh=0.25, max_iters=50):
    
#     "greedy inspired ransac; loop through all corners in each quadrant, compute angles, and pick the "
#     "set of points that best matches a rectangle"
    
#     candidates = np.array(candidates, dtype=np.float32)
#     corners = np.array(init_corners, dtype=np.float32)

#     best_score = rectangle_score(corners)
    
#     for _ in range(max_iters):

#         improved = False

#         for quad, (x0, x1, y0, y1) in enumerate(quadrants):
#             quad_pts = candidates[
#                 (candidates[:, 0] >= x0) & (candidates[:, 0] < x1) &
#                 (candidates[:, 1] >= y0) & (candidates[:, 1] < y1)
#             ]

#             if len(quad_pts) == 0:
#                 continue

#             current = corners[quad]
#             best_quad_score = best_score
#             best_quad_point = current

#             for p in quad_pts:

#                 corners[quad] = p
#                 score = rectangle_score(corners)

#                 if score < best_quad_score:
#                     best_quad_score = score
#                     best_quad_point = p
  
#             corners[quad] = best_quad_point

#             if best_quad_score < best_score:
#                 best_score = best_quad_score
#                 improved = True

#         if best_score <= score_thresh:
#             break

#         if not improved:
#             break

#     return None if best_score > score_thresh else corners


def detect_frame_corners(image, index):

    "Goal: find the four corners matching the corners of the rectangular frame for a honeycomb"
    "Design: "
    "1. Split the image into four local quadrants. "
    "2. Within each quadrant, find the corner that is closest to the corner of the picture frame"
    "3. Mark those foud corners for explainabilit (for now using a red x)"
    "4. return a list of those corners"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

    Ix, Iy = compute_gradients(gray)

    h_score = harris_cornerness_score(Ix, Iy)

    from skimage.feature import peak_local_max

    from sklearn.cluster import KMeans # added

    m_dist = 1 # reminder: increasees makes it less sensitive, decreases makes it more sensitive
    thresh_rel = 0.001

    coords = peak_local_max(h_score, min_distance=m_dist, threshold_rel=thresh_rel)

    edges_and_corners = coords[:, ::-1].astype(np.float32) 

    show_edges(gray, edges_and_corners, index)

    height, width = gray.shape

    half_width = width//2
    half_height = height//2


    filtered, approx_lines = filter_corners_and_edges_close_to_approximate_frame_edge(edges_and_corners, half_height, half_width, width, height)
    # # approximate border edges: 
    # x_min, y_min = edges_and_corners[:, 0].min(), edges_and_corners[:, 1].min()
    # x_max, y_max = edges_and_corners[:, 0].max(), edges_and_corners[:, 1].max()

    # def point_line_distance(pts, a, b, c):
    #     """distance from points to line ax + by + c = 0"""
    #     return np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / np.sqrt(a**2 + b**2)
    
    # top_dist    = point_line_distance(edges_and_corners, 0, 1, -y_min)
    # bottom_dist = point_line_distance(edges_and_corners, 0, 1, -y_max)
    # left_dist   = point_line_distance(edges_and_corners, 1, 0, -x_min)
    # right_dist  = point_line_distance(edges_and_corners, 1, 0, -x_max)

    # line_threshold = 50  # pixels — adjust as needed

    # min_dist = np.minimum(np.minimum(top_dist, bottom_dist),
    #                       np.minimum(left_dist, right_dist))
    
    # # filter the corners/edges cllse to the approximated lines:
    # filtered = edges_and_corners[min_dist < line_threshold]
    # print(f"Candidates after line filtering: {len(filtered)}")

    # if len(filtered) < 4:
    #     print("Not enough candidates after filtering, falling back")
    #     filtered = edges_and_corners

    candidates = []

    # for x, y in edges_and_corners:
    #     if is_corner(Ix, Iy, int(x), int(y)):
    #         candidates.append([x, y])
    for x, y in filtered:
        if is_corner(Ix, Iy, int(x), int(y)):
            candidates.append([x, y])

    candidates = np.array(candidates, dtype=np.float32)

    print(f"Number of candidates found: {len(candidates)}") 

    # find shape of the image to split into quadrants:


    image_corners = [np.array([0, 0]), np.array([width, 0]), 
                     np.array([0, height]), np.array([width, height])]
                    # in order: top left, top right, bottom left, bottom right

    quadrants = [(0, half_width, 0, half_height), (half_width, width, 0, half_height), (0, half_width, half_height, height), (half_width, width, half_height, height) ]

    selected_corners = []

    if len(candidates) < 4:
        print("Retake picture, not enough corners found")
        return None
    
    valid_quads = []

    for i, ((quad_left_side, quad_right_side, quad_bottom, quad_top), target) in enumerate(zip(quadrants, image_corners)):
        quad_coords = candidates[(candidates[:, 0] >= quad_left_side) & (candidates[:, 0] < quad_right_side) & (candidates[:, 1] >= quad_bottom) & (candidates[:, 1] < quad_top)] # find the cooridnates in repsective quad

        if len(quad_coords) == 0:
            valid_quads.append(False)
            continue
        else:
            valid_quads.append(True)

        distance_from_target = np.linalg.norm(quad_coords - target, axis=1) # normalize respective coordinates, and calcualte the respective distance from the image corners

        closest_corner = quad_coords[np.argmin(distance_from_target)]

        selected_corners.append(closest_corner) # append the closes corner

    corners = np.array(selected_corners, dtype=np.float32)


    # corners = refine_rectangle_ransac(
    #     candidates=candidates,
    #     init_corners=corners,
    #     quadrants=quadrants,
    #     score_thresh=0.25,
    #     max_iters=50
    # )


    draw_x(image, corners, gray, candidates, valid_quads, quadrants, index, approx_lines)
    

    return corners


# def ransac_select_corners(candidates, h_score, image_shape,
#                           num_iters=200, dist_weight=1.0, score_weight=1.0):
    
#     "RANSAC-inspired selection of 4 corners to filter noisy candidates."
    

#     H, W = image_shape

#     # ideal corner targets
#     targets = np.array([
#         [0, 0],
#         [W, 0],
#         [0, H],
#         [W, H]
#     ], dtype=np.float32)

#     best_score = -np.inf
#     best_corners = None

#     if len(candidates) < 4:
#         return None

#     candidates = np.asarray(candidates)

#     for _ in range(num_iters):

#         # ---- Step 1: random 4-point hypothesis ----
#         idx = np.random.choice(len(candidates), 4, replace=False)
#         sample = candidates[idx]

#         # ---- Step 2: assign each sampled point to closest target ----
#         used = set()
#         hypothesis = []

#         for t in targets:
#             dists = np.linalg.norm(sample - t, axis=1)

#             # penalize reused points
#             for j in range(len(dists)):
#                 if j in used:
#                     dists[j] = 1e9

#             best_j = np.argmin(dists)
#             used.add(best_j)
#             hypothesis.append(sample[best_j])

#         hypothesis = np.array(hypothesis)

#         # ---- Step 3: compute score ----

#         # geometric consistency (distance to ideal corners)
#         geom_error = np.linalg.norm(hypothesis - targets, axis=1).sum()

#         # corner strength (from Harris score map)
#         strength = 0
#         for x, y in hypothesis:
#             x, y = int(x), int(y)
#             if 0 <= x < W and 0 <= y < H:
#                 strength += h_score[y, x]

#         score = score_weight * strength - dist_weight * geom_error

#         # ---- Step 4: keep best ----
#         if score > best_score:
#             best_score = score
#             best_corners = hypothesis

#     return best_corners


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

def compute_homography(corners, width, height):

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


def warp(image, corners):

    x_coords = corners[:, 0]
    y_coords = corners[:, 1]

    min_width, max_width = np.min(x_coords), np.max(x_coords)
    min_height, max_height = np.min(y_coords), np.max(y_coords)

    width = int(max_width - min_width)
    height = int(max_height - min_height) 

    # above code is meant for approximating the dimensions of the found honeycomb in the image for better/mroe accurate image rectificaion

    H = compute_homography(corners, width, height)

    output_size: Tuple[int, int] = (width, height)

    # rectified = warp_helper()
    rectified = cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR)

    print(f"After warp - pixel at (0, 0): {rectified[0, 0]}")
    print(f"After warp - pixel at (0, 0): {rectified[0, 0]}")

    return rectified, H

def rectify_frame(image, index):

    corners = detect_frame_corners(image, index)

    print(f"corners: {corners}")

    if corners is None:
        # print("Corners is None")
        raise ValueError("PLEASE RETAKE IMAGE, CORNERS NOT DETECTED")
        # return None

    if not validation_check(corners, image, index):
        print("PLEASE RETAKE PHOTO AT A BETTER ANGLE — corners are inconsistent")


    rectified, H = warp(image, corners)

    print(f"rectified shape: {rectified.shape}")

    return rectified, H

