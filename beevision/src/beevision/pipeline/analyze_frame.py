import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import cv2

from geometry.rectification import rectify_frame


def analyze_frame(image_path: str):
    image = cv2.imread(image_path)

    rectified, H = rectify_frame(image, output_size=(1024, 1536))

    out_dir = Path("/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/interim/rectified") # save theim ages in interim for future segmentation
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "rectified_frame.jpg"
    cv2.imwrite(str(out_path), rectified)


    # cv2.imwrite("data/processed/rectified_frame_photo.jpg", rectified)

    # return {
    #     "rectified_image": rectified,
    #     "homography": H,
    # }

# def analyze_frame(image_path):
    
#     image = load_image(image_path)

#     rectified = rectify_frame(image)

#     cell_segmentation = segment_cells(rectified)

#     mite_detections = detect_mites(rectified)

#     health_metrics = compute_health_metrics(
#         segmentation=cell_segmentation,
#         mites=mite_detections,
#     )

#     health_score = compute_health_score(health_metrics)

#     return {
#         "rectified_image": rectified,
#         "cell_segmentation": cell_segmentation,
#         "mite_detections": mite_detections,
#         "metrics": health_metrics,
#         "health_score": health_score,
#     }

# test:
if __name__ == "__main__":
    path = "/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/raw/deepbee-classification/BEE_HOPE GIMONDE 2016_03_23 BL1_G FILE0658.JPG"
    analyze_frame(path) 
