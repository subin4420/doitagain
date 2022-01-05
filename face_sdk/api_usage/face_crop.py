"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import sys
sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
import cv2

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

if __name__ == '__main__':
    image_path = 'api_usage/test_images/test1.jpg'
    image_info_file = 'api_usage/test_images/test1_landmark_res0.txt'
    line = open(image_info_file).readline().strip()
    landmarks_str = line.split(' ')
    print(landmarks_str)
    landmarks = [float(num) for num in landmarks_str]
    
    face_cropper = FaceRecImageCropper()
    image = cv2.imread(image_path)
    cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
    cv2.imwrite('api_usage/temp/test1_cropped.jpg', cropped_image)
    logger.info('Crop image successful!')

if __name__ == 'face_crop':
    image_path = 'api_usage/hallway/merged.jpg'
    image_info_file = 'api_usage/hallway/test1_landmark_res0.txt'
    line = open(image_info_file).readline().strip()
    landmarks_str = line.split(' ')
    landmarks = [float(num) for num in landmarks_str]
    print(landmarks_str)

    face_cropper = FaceRecImageCropper()
    image = cv2.imread(image_path)
    cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
    cv2.imwrite('api_usage/hallway/test1_cropped.jpg', cropped_image)
    logger.info('Crop image successful!')
"""
    image_path2 = 'api_usage/test_images/test2.jpg'
    image_info_file2 = 'api_usage/test_images/test2_landmark_res0.txt'
    line = open(image_info_file2).readline().strip()
    landmarks_str2 = line.split(' ')
    landmarks2 = [float(num) for num in landmarks_str2]

    face_cropper2 = FaceRecImageCropper()
    image2 = cv2.imread(image_path2)
    cropped_image2 = face_cropper.crop_image_by_mat(image2, landmarks2)
    cv2.imwrite('api_usage/temp/test2_cropped.jpg', cropped_image2)
    logger.info('Crop image successful!')
"""
