import numpy as np
from scipy.signal import convolve2d 
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt

def gaussian_filter(image, sigma) -> int:
    kernel_size = (int)(6*sigma+1)
    k = kernel_size // 2
    filter = np.arange(-k, k+1)
    gaussian_filter = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(filter**2)/(2*sigma**2))
    gaussian_filter /= gaussian_filter.sum()
    smoothed = convolve2d(image, gaussian_filter.reshape(1, -1), mode='same') 
    smoothed = convolve2d(smoothed, gaussian_filter.reshape(-1, 1), mode='same')
    return smoothed

def cornerness(Ix2, Iy2, Ixy, alpha, sigma):
    gIx2 = gaussian_filter(Ix2, sigma)
    gIy2 = gaussian_filter(Iy2, sigma)
    gIxy = gaussian_filter(Ixy, sigma)
    cornerness_score = gIx2 * gIy2 - gIxy**2 - alpha*(gIx2 + gIy2)**2
    return cornerness_score


def compute_gradients(gray):
    Ix = np.zeros_like(gray)
    Iy = np.zeros_like(gray)
    Ix[:, :-1] = gray[:, 1:] - gray[:, :-1]
    Iy[:-1, :] = gray[1:, :] - gray[:-1, :]
    return Ix, Iy

def harris_cornerness_score(Ix, Iy):
    Ix2, Iy2, Ixy = Ix**2, Iy**2, Ix*Iy
    h_score = cornerness(Ix2, Iy2, Ixy, alpha=0.05, sigma=1)
    return h_score


def warp_helper():

    return

def rectangle_score(corners):
    corners = np.array(corners, dtype=np.float32)
    if len(corners) != 4:
        return 1e9

    s = corners.sum(axis=1)
    d = corners[:, 0] - corners[:, 1]

    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]
    tr = corners[np.argmax(d)]
    bl = corners[np.argmin(d)]

    ordered = np.array([tl, tr, br, bl])

    def compute_angle_between_three_points(a, b, c):
        ba = a - b
        bc = c - b
        return np.arccos( np.clip( np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6), -1, 1))

    angles = [
        compute_angle_between_three_points(ordered[3], ordered[0], ordered[1]),
        compute_angle_between_three_points(ordered[0], ordered[1], ordered[2]),
        compute_angle_between_three_points(ordered[1], ordered[2], ordered[3]),
        compute_angle_between_three_points(ordered[2], ordered[3], ordered[0]),
    ]

    angle_error = np.mean(np.abs(np.array(angles) - np.pi/2))
    return angle_error


def fit_line_regression(band_pts, horizontal=True):
    """
    fit y = mx + b (horizontal=True) or x = my + b (horizontal=False)
    returns (a, b, c) for line ax + by + c = 0
    """
    if len(band_pts) < 2:
        return None
    if horizontal:
        # fit y = mx + b → predict y from x
        X = band_pts[:, 0].reshape(-1, 1)
        Y = band_pts[:, 1]
    else:
        # fit x = my + b → predict x from y
        X = band_pts[:, 1].reshape(-1, 1)
        Y = band_pts[:, 0]

    reg = LinearRegression().fit(X, Y)
    m = reg.coef_[0]
    b = reg.intercept_

    if horizontal:
        # y = mx + b → mx - y + b = 0
        a_coef, b_coef, c_coef = m, -1, b
    else:
        # x = my + b → -x + my + b = 0
        a_coef, b_coef, c_coef = -1, m, b

    norm = np.sqrt(a_coef**2 + b_coef**2)
    return a_coef / norm, b_coef / norm, c_coef / norm


def point_line_distance(pts, line, ):
    if line is None:
        return np.full(len(pts), np.inf)
    a, b, c = line
    return np.abs(a * pts[:, 0] + b * pts[:, 1] + c)


def average_lines(*lines):
    """average multiple lines (a,b,c) together, ignoring Nones"""
    valid = [l for l in lines if l is not None]
    if len(valid) == 0:
        return None
    avg = np.mean(valid, axis=0)
    # renormalize
    norm = np.sqrt(avg[0]**2 + avg[1]**2)
    return tuple(avg / norm)

def jittered_quadrants(width, height, half_w, half_h, jitter=0.1):
    """
    jitter = fraction of width/height to perturb boundaries
    """

    dx = int(width * jitter)
    dy = int(height * jitter)

    cx = half_w + np.random.randint(-dx, dx+1)
    cy = half_h + np.random.randint(-dy, dy+1)

    return [
        (0, cx, 0, cy),        # TL
        (cx, width, 0, cy),    # TR
        (0, cx, cy, height),   # BL
        (cx, width, cy, height)# BR
    ]


def random_quads(width, height, half_width, half_height, pts, top_lines, bottom_lines, left_lines,right_lines):
    for i in range(20):  # randomize the iterations
        quad = jittered_quadrants(width, height, half_width, half_height, jitter=0.08)

        (_, xmax1, _, ymax1) = quad[0]
        (xmin2, _, _, ymax2) = quad[1]
        (_, xmax3, ymin3, _) = quad[2]
        (xmin4, _, ymin4, _) = quad[3]

        x = pts[:, 0]
        y = pts[:, 1]

        tl = pts[(x < xmax1) & (y < ymax1)]
        tr = pts[(x >= xmin2) & (y < ymax2)]
        bl = pts[(x < xmax3) & (y >= ymin3)]
        br = pts[(x >= xmin4) & (y >= ymin4)]

        if len(tl) > 2:
            top_lines.append(fit_line_regression(tl, horizontal=True))
        if len(tr) > 2:
            top_lines.append(fit_line_regression(tr, horizontal=True))

        if len(bl) > 2:
            bottom_lines.append(fit_line_regression(bl, horizontal=True))
        if len(br) > 2:
            bottom_lines.append(fit_line_regression(br, horizontal=True))

        # vertical
        if len(tl) > 2:
            left_lines.append(fit_line_regression(tl, horizontal=False))
        if len(bl) > 2:
            left_lines.append(fit_line_regression(bl, horizontal=False))

        if len(tr) > 2:
            right_lines.append(fit_line_regression(tr, horizontal=False))
        if len(br) > 2:
            right_lines.append(fit_line_regression(br, horizontal=False))



def filter_corners_and_edges_close_to_approximate_frame_edge(edges_and_corners, half_height, half_width, width, height):
    "Purpose: do two passes for linear regression with differetn orientations. We do this to approxiamte the edges of "
    "the rectangualr beehive frame to filter out noise in the image. "

    "Design:"
    "1. In first pass break image into triangular quads. Find lineear regression for each of those quads"
    "2. in second pass break image into rectangualr/square quads. Find linear regression for each of those quads"
    "3. average between the linear regressions in each pass:"
    "   a. top quad in first pass averages with top tow quads in second pass"
    "   b. right quad in first pass averages with two right quads in second pass"
    "   c. bottom quad in first pass averages with two bottom quads in second pass"
    "   d. left quad in first pass averages with two left wuads in second pass"
    "4. Filter corenrs close to the final linear regression"

    pts = edges_and_corners
    x = pts[:, 0]
    y = pts[:, 1]

    top_triangle   = (y < half_height) & (np.abs(x - half_width) < y)
    bottom_triangle = (y > half_height) & (np.abs(x - half_width) < (height - y))
    left_triangle   = (x < half_width) & (np.abs(y - half_height) < x)
    right_triangle  = (x > half_width) & (np.abs(y - half_height) < (width - x))

    top_line    = fit_line_regression(pts[top_triangle],    horizontal=True)
    bottom_line = fit_line_regression(pts[bottom_triangle], horizontal=True)
    left_line   = fit_line_regression(pts[left_triangle],   horizontal=False)
    right_line  = fit_line_regression(pts[right_triangle],  horizontal=False)

    # stpe 1: first pass (triangualr quads):

    # make linear regression in each quad to approximate frame edges:
    # top_pts    = pts[top_triangle]
    # bottom_pts = pts[bottom_triangle]
    # left_pts   = pts[left_triangle]
    # right_pts  = pts[right_triangle]

    # top_line    = fit_line_regression(top_pts,    horizontal=True)
    # bottom_line = fit_line_regression(bottom_pts, horizontal=True)
    # left_line   = fit_line_regression(left_pts,   horizontal=False)
    # right_line  = fit_line_regression(right_pts,  horizontal=False)

    # # step 2: second pass (rectangualr/square quads)

    top_h    = int(height * 0.25)
    bottom_h = int(height * 0.75)
    left_w   = int(width  * 0.25)
    right_w  = int(width  * 0.75)

    top_left_line   = fit_line_regression(pts[(y < top_h)     & (x < half_width)],  horizontal=True)
    top_right_line  = fit_line_regression(pts[(y < top_h)     & (x >= half_width)], horizontal=True)
    bot_left_line   = fit_line_regression(pts[(y >= bottom_h) & (x < half_width)],  horizontal=True)
    bot_right_line  = fit_line_regression(pts[(y >= bottom_h) & (x >= half_width)], horizontal=True)
    left_top_line   = fit_line_regression(pts[(x < left_w)    & (y < half_height)], horizontal=False)
    left_bot_line   = fit_line_regression(pts[(x < left_w)    & (y >= half_height)],horizontal=False)
    right_top_line  = fit_line_regression(pts[(x >= right_w)  & (y < half_height)], horizontal=False)
    right_bot_line  = fit_line_regression(pts[(x >= right_w)  & (y >= half_height)],horizontal=False)

    # top_h    = int(height * 0.25)   # top 25% of image
    # bottom_h = int(height * 0.75)   # bottom 25% starts here
    # left_w   = int(width  * 0.25)   # left 25% of image
    # right_w  = int(width  * 0.75)   # right 25% starts here


    # # top strip: split into left, center, right
    # top_left_mask   = (y < top_h) & (x < half_width)
    # top_right_mask  = (y < top_h) & (x >= half_width)

    # # bottom strip: split into left, center, right
    # bot_left_mask   = (y >= bottom_h) & (x < half_width)
    # bot_right_mask  = (y >= bottom_h) & (x >= half_width)

    # # left strip: split into top, bottom
    # left_top_mask   = (x < left_w) & (y < half_height)
    # left_bot_mask   = (x < left_w) & (y >= half_height)

    # # right strip: split into top, bottom
    # right_top_mask  = (x >= right_w) & (y < half_height)
    # right_bot_mask  = (x >= right_w) & (y >= half_height)

    # top_left_pts   = pts[top_left_mask]
    # top_right_pts  = pts[top_right_mask]
    # bot_left_pts   = pts[bot_left_mask]
    # bot_right_pts  = pts[bot_right_mask]
    # left_top_pts   = pts[left_top_mask]
    # left_bot_pts   = pts[left_bot_mask]
    # right_top_pts  = pts[right_top_mask]
    # right_bot_pts  = pts[right_bot_mask]

    # top_left_line   = fit_line_regression(top_left_pts,   horizontal=True)
    # top_right_line  = fit_line_regression(top_right_pts,  horizontal=True)
    # bot_left_line   = fit_line_regression(bot_left_pts,   horizontal=True)
    # bot_right_line  = fit_line_regression(bot_right_pts,  horizontal=True)
    # left_top_line   = fit_line_regression(left_top_pts,   horizontal=False)
    # left_bot_line   = fit_line_regression(left_bot_pts,   horizontal=False)
    # right_top_line  = fit_line_regression(right_top_pts,  horizontal=False)
    # right_bot_line  = fit_line_regression(right_bot_pts,  horizontal=False)

    # step 3: average tbetween the passes
    top_lines    = [top_line,    top_left_line,  top_right_line]
    bottom_lines = [bottom_line, bot_left_line,  bot_right_line]
    left_lines   = [left_line,   left_top_line,  left_bot_line]
    right_lines  = [right_line,  right_top_line, right_bot_line]

    # random jittered quads appends more lines to each list
    random_quads(width, height, half_width, half_height, pts, top_lines, bottom_lines, left_lines, right_lines)
                 
    # top_final    = average_lines(top_line, top_left_line, top_right_line)

    # bottom_final = average_lines(bottom_line, bot_left_line, bot_right_line)
    # left_final   = average_lines(left_line, left_top_line, left_bot_line)
    # right_final  = average_lines(right_line, right_top_line, right_bot_line)



    # random_quads(width, height, half_width, half_height, pts, top_final, bottom_final, left_final,right_final)


    top_final    = average_lines(*top_lines)
    bottom_final = average_lines(*bottom_lines)
    left_final   = average_lines(*left_lines)
    right_final  = average_lines(*right_lines)

    # step 4: filter points close to frame edges:
    line_threshold = 20  

    d_top    = point_line_distance(pts, top_final)
    d_bottom = point_line_distance(pts, bottom_final)
    d_left   = point_line_distance(pts, left_final)
    d_right  = point_line_distance(pts, right_final)
    # d_top    = point_line_distance(pts, top_line)
    # d_bottom = point_line_distance(pts, bottom_line)
    # d_left   = point_line_distance(pts, left_line)
    # d_right  = point_line_distance(pts, right_line)

    min_dist = np.minimum(np.minimum(d_top, d_bottom),
                          np.minimum(d_left, d_right))

    filtered = pts[min_dist < line_threshold]
    print(f"Candidates after line filtering: {len(filtered)}")

    if len(filtered) < 4:
        print("Not enough candidates after filtering, falling back")
        filtered = pts

    # approx_lines = {
    #     'top': top_line,
    #     'bottom': bottom_line,
    #     'left': left_line,
    #     'right': right_line,
    #     'width': width,
    #     'height': height,
    # }
    approx_lines = {
        'top': top_final,
        'bottom': bottom_final,
        'left': left_final,
        'right': right_final,
        'width': width,
        'height': height,
    }

    return filtered, approx_lines


def show_edges(gray, edges_and_corners, index):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(gray, cmap='gray')
    ax.plot(edges_and_corners[:, 0], edges_and_corners[:, 1], 
            '.b', markersize=3, label=f'all found ({len(edges_and_corners)})')
    ax.set_title(f"All found edges and corners: {len(edges_and_corners)}")
    ax.axis('off')
    ax.legend()
    debug_path = f"/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/src/beevision/data/interim/marked_corners/all_found_{index}.png"
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    plt.savefig(debug_path)
    plt.close()
    print(f"Saved all found corners to {debug_path}")


# def get_feature_descriptors(image, xs, ys, window_width, mode, image_file=None):
#     '''
#     Computes a feature descriptor for each feature point.

#     Implement two modes (use the `mode` argument to toggle):
#       "patch" — simple image patch descriptor
#       "sift"  — SIFT-like gradient histogram descriptor

#     Compare to a third mode using state of the art features
#     No implementation necessary:
#       "dinov3" - self-supervised deep learned generic features

#     IMAGE PATCH:
#       1. Cut out a window_width x window_width patch around each point.
#       2. Flatten to a 1-d vector and normalize to unit length.

#     SIFT (see Lowe, http://www.cs.ubc.ca/~lowe/keypoints/):
#       1. Compute image gradients (magnitude and orientation).
#       2. For each point, divide the window into a 4x4 grid of cells
#          (each cell is window_width/4 pixels).
#       3. In each cell, bin gradient magnitudes into 8 orientation bins.
#       4. Concatenate all histograms → 4x4x8 = 128-d vector.
#       5. Normalize to unit length.

#     Optional enhancements for better performance:
#       - Interpolate contributions across neighboring cells and bins.
#       - Normalize → threshold at 0.2 → re-normalize (reduces lighting effects).
#       - Raise elements to a power < 1 (e.g., sqrt) for robustness.

#     :params:
#     :image: a grayscale or color image (your choice depending on your implementation)
#     :xs: np.array of x coordinates (column indices) of feature points
#     :ys: np.array of y coordinates (row indices) of feature points
#     :window_width: in pixels, is the local window width (always a multiple of 4).
#     :mode: "patch", "sift", or "dinov3"
#     :image_file: (optional) path to the image file, used for DINOv3 cache lookup

#     :returns:
#     :features: np.array of shape (len(xs), feature_dim). For SIFT, feature_dim = 128.
#     '''
#     if mode == "patch":
#         # TODO: Your implementation here!
#         # These are placeholders - replace with your feature descriptors!
#         features = []
#         width = window_width // 2
#         for x, y in zip(xs, ys):
#             patch = image[y - width: y + width, x - width: x + width]
#             if patch.shape != (window_width, window_width):
#                 continue
#             flattened_patch = patch.flatten()
#             norm = np.sqrt(np.sum(flattened_patch**2))+ 1e-10
#             # norm = np.linalg.norm(flattened_patch, axis=1, keepdims=True)
#             normalized_patch = flattened_patch / norm
#             features.append(normalized_patch)
#         features = np.array(features)

#     elif mode == "sift":
#         # TODO: Your implementation here!
#         # These are placeholders - replace with your feature descriptors!
#         Ix = np.gradient(image, axis = 1)
#         Iy = np.gradient(image, axis = 0)
#         mag = np.sqrt(Ix**2 + Iy**2)
#         orient = np.degrees(np.arctan2(Iy, Ix)) % 360

#         cell_width = window_width // 4

#         features = []
#         width = window_width // 2

#         print("xs: " + str(np.shape(xs)))     
        
#         for x, y in zip(xs, ys):
#             patch_mag = mag[y - width: y + width, x - width: x + width]
#             patch_orient = (orient[y - width: y + width, x - width: x + width] // 45)
#             if np.shape(patch_mag) != (window_width, window_width):
#                 continue  
#             patch_orient = patch_orient.reshape(4, cell_width, 4, cell_width)
#             patch_mag = patch_mag.reshape(4, cell_width, 4, cell_width)

#             histogram = np.zeros((4,4,8))
#             for i in range(8):
#                 valid_orient = patch_orient == i #logical indexing!!! Yayy!
#                 histogram[:,:,i] = np.sum(valid_orient*patch_mag, axis=(1,3))
#             descriptor = histogram.reshape(128)
#             features.append(descriptor)
#         features = np.array(features)
#         norm = np.sqrt(np.sum(features**2, axis=1, keepdims=True))+ 1e-10
#         # norm = np.linalg.norm(features, axis=1, keepdims=True)
#         features = np.array(features) / norm

#     elif mode == "dinov3":
#         # DINOv3 is handled here — you don't need to implement it.
#         cache_path = os.path.splitext(image_file)[0] + "_dinov3.npz" if image_file else None
#         fmap, meta = compute_dino_feature_map(image, cache_path=cache_path)
#         features = sample_dino_descriptors(fmap, meta, xs, ys)

#     return features
    