import cv2
from geometry.helpers import compute_gradients, harris_cornerness_score, warp_helper, rectangle_score, fit_line_regression, point_line_distance, filter_corners_and_edges_close_to_approximate_frame_edge, show_edges, invalid_print, gaussian_filter
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import os


def draw_x(image, corners, gray, candidates, valid_quads, quadrants, index, approx_lines):
    "marks all found corners iwth a green circle, the chosen red corners wth a red x, and illustrates if quad was abel to find a corner "
    "or not for explainability/visibility"

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

    # draw the lines for visibiltiy of linear regression from approximate lineslines
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

    debug_path = f"/Users/ethanlin/Comb-puter-Vision/interim/marked_corners/final_four_corners_{index}.png"
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    plt.savefig(debug_path)
    plt.close()

    print(f"Saved debug to {debug_path}")


def validation_check(corners, image, index):
    "checks if detected corners form a consistent rectangle"

    
    if len(corners) < 4:
        return None, False

    # order corners first so we know which is which
    ordered = order_corners(corners)
    if ordered is None:
        return None, False

    top_left = ordered[0] 
    top_right = ordered[1]  
    bottom_right = ordered[2]  
    bottom_left = ordered[3]  

    left_height  = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)
    top_width    = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)

    coord_height = max(left_height, right_height)
    coord_width = max(top_width, bottom_width)
    height_threshold = coord_height * 0.25
    width_threshold  = coord_width * 0.25

    error = False

    height_diff = abs(left_height - right_height)
    print(f"Height difference: {height_diff:.1f} (threshold: {height_threshold})")
    if height_diff > height_threshold:
        error = True

    width_diff = abs(top_width - bottom_width)
    print(f"Width difference:  {width_diff:.1f} (threshold: {width_threshold})")
    if width_diff > width_threshold:
        error = True

    if error == False:
        print("Validation passed!")
        return None, True

    # draw warning on image and save
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ax.imshow(gray, cmap='gray')

    ordered = np.array([top_left, top_right, bottom_right, bottom_left])
    ax.plot(ordered[:, 0], ordered[:, 1], '+r', markersize=20)

    # draw sides colored by pass/fail
    sides = [ (top_left, top_right, f"top: {top_width:.0f}px",       width_diff  <= width_threshold), 
             (bottom_left, bottom_right, f"bottom: {bottom_width:.0f}px",  width_diff  <= width_threshold),
             (top_left, bottom_left, f"left: {left_height:.0f}px",     height_diff <= height_threshold),
             (top_right, bottom_right, f"right: {right_height:.0f}px",   height_diff <= height_threshold),
    ]

    for p1, p2, label, passed in sides:
        color = 'green' if passed else 'red'
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=3)
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        ax.text(mid_x, mid_y, label, color=color, fontsize=10,
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    warning_text = "Please retake photo at better angle\n"
    ax.text(0.5, 0.05, warning_text, transform=ax.transAxes,
            fontsize=12, color='red', ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax.set_title("Validation for corners failed, please retake photo", color='red', fontsize=14)
    ax.axis('off')

    debug_path = f"/Users/ethanlin/Comb-puter-Vision/interim/marked_corners/validation_fail_{index}.png"
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    plt.savefig(debug_path)


    error_rectified_path = f"/Users/ethanlin/Comb-puter-Vision/interim/rectified/retake_image_bad_angle_{index}.png"
    os.makedirs(os.path.dirname(error_rectified_path), exist_ok=True)
    plt.savefig(error_rectified_path)
    plt.close()

    print(f"validation errored:")
    print(f"Saved validation fail image to {debug_path}")


    new_rectified_error_img = f"/Users/ethanlin/Comb-puter-Vision/interim/rectified/validation_fail_{index}.png"

    return new_rectified_error_img, False

    

def is_corner(Ix, Iy, x, y, window=3, ratio_thresh=0.05):

    Ix2 = Ix[y - window: y + window +1, x-window: x + window +1] **2

    Iy2 = Iy[y - window: y + window +1, x -window: x +window+ 1] **2

    Ixy = Ix[y - window: y + window +1, x-window: x +window +1] * Iy[y-window: y + window + 1, x-window: x + window + 1]

    Sxx = Ix2.sum()

    Syy = Iy2.sum()

    Sxy = Ixy.sum()

    M = np.array([[Sxx, Sxy], [Sxy, Syy]])


    eigvals = np.linalg.eigvalsh(M) 

    l1, l2 = eigvals[1], eigvals[0]

    ratio = l2 / (l1 + 1e-6)

    return ratio > ratio_thresh


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

    m_dist = 1 # reminder: increasees makes it less sensitive, decreases makes it more sensitive
    thresh_rel = 0.001

    coords = peak_local_max(h_score, min_distance=m_dist, threshold_rel=thresh_rel)

    edges_and_corners = coords[:, ::-1].astype(np.float32) 

    show_edges(gray, edges_and_corners, index)

    height, width = gray.shape

    half_width = width//2
    half_height = height//2


    filtered, approx_lines = filter_corners_and_edges_close_to_approximate_frame_edge(edges_and_corners, half_height, half_width, width, height)

    candidates = []

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

        selected_corners.append(closest_corner) # append the closest corner

    corners = np.array(selected_corners, dtype=np.float32)

    draw_x(image, corners, gray, candidates, valid_quads, quadrants, index, approx_lines)
    

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
    print(f"  top -right:    {ordered[1]}")
    print(f"  bottom-right: {ordered[2]}")
    print(f"  bottom-left:  {ordered[3]}")

    return ordered


def check_reporojection(src, dst, H):
    print("reprojection check:")
    for i in range(4):
        x, y = src[i]
        pt = H @ np.array([x, y, 1])
        pt = pt / pt[2] 
        print(f" source {src[i]}, projected {pt[:2]}, expected { dst[i]}")

def compute_homography(corners, width, height):

    src = order_corners(corners)

    dst = np.array( [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    print("destination corners:")
    print(f"  top - left:     {dst[0]}")
    print(f"  top- right:    {dst[1]}")
    print(f"  bottom -right: {dst[2]}")
    print(f"  bottom -left:  {dst[3]}")

    A = []
    for i in range(4):
        x, y = src[i]
        xp, yp = dst[i]
        A.append([-x, - y, -1,  0,  0 ,  0,  xp * x,  xp * y,  xp])

        A.append([ 0,  0,  0, -x, -y, -1,  yp * x,  yp * y,  yp])
    
    A = np.array(A)

    _, _, right_singluar_vectors = np.linalg.svd(A) # use svd to extract right singualr vectors
    
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

    # above code is meant for approximating the dimensions of the found honeycomb in the image for better/mroe accurate image rectificaion, closer tothe dimensions of the honeycomb

    H = compute_homography(corners, width, height)

    output_size: Tuple[int, int] = (width, height)

    rectified = cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR)

    print(f"after warp,  pixel at (0, 0): {rectified[0, 0]}")
    print(f"After warp, pixel at (0, 0): {rectified[0, 0]}")

    return rectified, H

def rectify_frame(image, index):

    corners = detect_frame_corners(image, index)

    print(f"corners: {corners}")

    if corners is None:
        raise ValueError("Please retake photo, corners not detected")")

    error_image, passed = validation_check(corners, image, index)

    if not passed:
        return error_image, None
        
    rectified, H = warp(image, corners)

    print(f"rectified shape: {rectified.shape}")

    return rectified, H

