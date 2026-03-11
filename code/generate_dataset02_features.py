import numpy as np
import cv2
from feature_detect_match import feature_detect_match

if __name__ == "__main__":

    raw_imgs_path = "../data/dataset02/dataset02_imgs.npy"
    
    print("Reading dataset02_imgs.npy")

    try:
        imgs_data = np.load(raw_imgs_path, allow_pickle=True).item()
        left_imgs = imgs_data['cam_imgs_L']
        right_imgs = imgs_data['cam_imgs_R']
    except Exception as e:
        print(f"failed")
        exit()

    T_left = len(left_imgs)
    T_right = len(right_imgs)
    print(f"Total left frrame: {T_left}")
    print(f"Total rigtt frrame: {T_right}")
    
    # grey
    if left_imgs[0].ndim == 3:
        left_imgs = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in left_imgs]
        right_imgs = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in right_imgs]

    print("executing feature_detect_match")
    z_t = feature_detect_match(left_imgs, right_imgs)
    
    output_path = "../data/dataset02/dataset02_features.npy"
    np.save(output_path, z_t)
    print(f"saved file: {output_path}")