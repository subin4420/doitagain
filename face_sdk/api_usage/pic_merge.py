import cv2
import numpy as np
import os

def my_merge(pic_path, pic1_name,pic2_name ):

    pic1_root_dir = os.path.join(pic_path, pic1_name)
    pic2_root_dir = os.path.join(pic_path, pic2_name)

    img1 = cv2.imread(pic1_root_dir, 1)
    img2 = cv2.imread(pic1_root_dir, 1)

# 이미지 붙이기
    addh = np.hstack((img1, img2))
    cv2.imwrite("api_usage/hallway/test1.jpg", addh)
    print("merge success")
    return addh


pic1_root_dir = os.path.join('C:/Users/kii/Desktop/FaceX-Zoo-main/face_sdk/api_usage/test_images', 'pic1')
pic2_root_dir = os.path.join('C:/Users/kii/Desktop/FaceX-Zoo-main/face_sdk/api_usage/test_images', 'pic2')
print(pic1_root_dir)
print(pic2_root_dir)
img1 = cv2.imread(pic1_root_dir, 1)
img2 = cv2.imread(pic1_root_dir, 1)

# 이미지 붙이기
addh = np.hstack((img1, img2))
cv2.imwrite("api_usage/hallway/test1.jpg", addh)
print("merge success")
