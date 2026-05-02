import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import cv2

from geometry.rectification import rectify_frame


def analyze_frame(image_path: str, index):
    
    image = cv2.imread(image_path)

    if image is None:
        print("no image loaded error")
        return

    rectified, H = rectify_frame(image, index)

    if rectified is None:
        print("rectified is none")
        return

    out_dir = Path("/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/src/beevision/data/interim/rectified") # save theim ages in interim for future segmentation
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"rectified_frame_{index}.jpg"
    # cv2.imwrite(str(out_path), rectified)
    result = cv2.imwrite(str(out_path), rectified)
    print(f"Write success: {result}")
    print(f"Saved to {out_path}")


# test:
if __name__ == "__main__":
    # path = "/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/test_images_rectification/control_test.jpg"
    path7 = "/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/test_images_rectification/honeycomb_test_7.jpg"
    path3 = "/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/test_images_rectification/tilted_beehive_3.jpg"
    path5 = "/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/test_images_rectification/tilted_beehive_5.jpg"
    path6 = "/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/test_images_rectification/tilted_beehive_6.jpg"
    path8 = "/Users/ethanlin/CSCI1430_Homeworks/Comb-puter-Vision/beevision/data/test_images_rectification/tilted_beehive_8.jpg"
    # analyze_frame(path, 1) 
    analyze_frame(path7, 7) 
    analyze_frame(path3, 3) 
    analyze_frame(path5, 5) 
    analyze_frame(path6, 6) 
    analyze_frame(path8, 8) 
    # analyze_frame(path3, 3) 
