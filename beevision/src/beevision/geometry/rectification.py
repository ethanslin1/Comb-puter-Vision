


def rectify_frame(image: np.ndarray) -> dict:
    """
    Input: raw frame image (H, W, 3)
    Output:
        {
            "warped": rectified image,
            "corners": 4x2 array,
            "homography": 3x3 matrix,
            "success": bool
        }
    """