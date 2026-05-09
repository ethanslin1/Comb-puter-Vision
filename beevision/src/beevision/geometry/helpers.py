import numpy as np
from scipy.signal import convolve2d 
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt
import cv2

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
    Ix2, Iy2, Ixy = Ix **2, Iy **2, Ix*Iy
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
        return np.arccos( np.clip( np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6), -1, 1))

    angles = [
        compute_angle_between_three_points(ordered[3], ordered[0], ordered[1]),
        compute_angle_between_three_points(ordered[0], ordered[1], ordered[2]),
        compute_angle_between_three_points(ordered[1], ordered[2], ordered[3]),
        compute_angle_between_three_points(ordered[2], ordered[3], ordered[0]),
    ]

    angle_error = np.mean(np.abs(np.array(angles) - np.pi/2))
    return angle_error


def fit_line_regression(band_pts, horizontal=True):
    " fit y = mx + b (horizontal=True) or x = my + b (horizontal=False)"
    " returns (a, b, c) for line ax + by + c = 0"
    if len(band_pts) < 2:
        return None
    if horizontal:
        X = band_pts[:, 0].reshape(-1, 1)
        Y = band_pts[:, 1]
    else:
        X = band_pts[:, 1].reshape(-1, 1)
        Y = band_pts[:, 0]

    reg = LinearRegression().fit(X, Y)
    m = reg.coef_[0]
    b = reg.intercept_

    if horizontal:
        a_coefficient = m
        b_coefficient = -1
        c_coefficient =  b
    else:
        a_coefficient = -1
        b_coefficient = m
        c_coefficient = b

    norm = np.sqrt(a_coefficient ** 2 + b_coefficient ** 2)
    return a_coefficient / norm, b_coefficient / norm, c_coefficient / norm


def point_line_distance(pts, line, ):
    if line is None:
        return np.full(len(pts), np.inf)
    a, b, c = line
    return np.abs(a * pts[:, 0] + b * pts[:, 1] + c)


def average_lines(*lines):
    "average multiple lines (a, b ,c) together, ignoring Nones"
    valid = [l for l in lines if l is not None]
    if len(valid) == 0:
        return None
    avg = np.mean(valid, axis=0)
    norm = np.sqrt(avg[0]**2 + avg[1]**2)
    return tuple(avg / norm)

def jittered_quadrants(width, height, half_w, half_h, jitter=0.1):
    "jitter = fraction of width / height to perturb boundaries"

    dx = int(width * jitter)
    dy = int(height * jitter)

    cx = half_w + np.random.randint(-dx, dx+1)
    cy = half_h + np.random.randint(-dy, dy+1)

    return [(0,cx,0, cy),(cx, width, 0, cy),(0,cx,cy, height),(cx, width, cy, height)]


def random_quads(width, height, half_width, half_height, pts, top_lines, bottom_lines, left_lines,right_lines):
    for i in range(20):  # randomize the  iterations
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

        # horizontal:
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

    # step 3: average tbetween the passes
    top_lines    = [top_line,    top_left_line,  top_right_line]
    bottom_lines = [bottom_line, bot_left_line,  bot_right_line]
    left_lines   = [left_line,   left_top_line,  left_bot_line]
    right_lines  = [right_line,  right_top_line, right_bot_line]

    random_quads(width, height, half_width, half_height, pts, top_lines, bottom_lines, left_lines, right_lines)

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

    min_dist = np.minimum(np.minimum(d_top, d_bottom),
                          np.minimum(d_left, d_right))

    filtered = pts[min_dist < line_threshold]
    print(f"vandidates after line filtering: {len(filtered)}")

    if len(filtered) < 4:
        print("not enough candidates from filter")
        filtered = pts

    approx_lines = {'top': top_final, 'bottom': bottom_final,'left': left_final,'right': right_final,'width': width,'height': height}

    return filtered, approx_lines


def show_edges(gray, edges_and_corners, index):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(gray, cmap  ='gray')

    ax.plot(edges_and_corners[:, 0], edges_and_corners[:, 1], 
            '.b', markersize=3, label=f' all found ({len(edges_and_corners)})')
    ax.set_title(f"all found edges and corners: {len(edges_and_corners)}")
    ax.axis('off')

    ax.legend()

    debug_path = f"/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/src/beevision/data/interim/marked_corners/all_found_{index}.png"

    os.makedirs(os.path.dirname(debug_path), exist_ok=True)

    plt.savefig(debug_path)
    plt.close()
    print(f"Saved all found corners to {debug_path}")




def invalid_print(image, index):
    print("Please take photo at better angle — corners are inconsistent")
            
    warning_image = image.copy()
    h, w = warning_image.shape[:2]
    
    # draw red rectangle border
    cv2.rectangle(warning_image, (0, 0), (w-1, h-1), (0, 0, 255), 20)
    
    # draw warning text
    text1 = "Retake photo"
    text2 = "Corners Inconsistent - Bad Angle"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 1000  # scale with image size
    thickness = max(2, int(font_scale * 3))
    
    (tw1, th1), _ = cv2.getTextSize(text1, font, font_scale * 2, thickness)
    (tw2, th2), _ = cv2.getTextSize(text2, font, font_scale, thickness)

    cv2.putText(warning_image, text1,
                (w//2 - tw1//2, h//2 - 20),
                font, font_scale * 2, (0, 0, 255), thickness)
    cv2.putText(warning_image, text2,
                (w//2 - tw2//2, h//2 + th2 + 20),
                font, font_scale, (0, 0, 255), thickness)

    # save it
    out_path = f"/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/interim/rectified/retake_required_{index}.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cv2.imwrite(out_path, warning_image)
    
    print(f"Saved retake warning image to {out_path}")

    return warning_image, None
