import numpy as np
from scipy.signal import convolve2d 

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
    