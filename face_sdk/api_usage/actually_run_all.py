import sys
import face_merge
sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import numpy as np
#-----------------detect-------------------
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
#-----------------align-------------------
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
#----------------crop---------------------
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
#----------------------feature----------------------
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler
with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.Loader)

if __name__ == '__main__':
    # common setting for all model, need not modify.
    model_path = 'models'
    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name = model_conf[scene][model_category]

    # load model
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)

    try:
        model, cfg = faceDetModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)

    # read image
    image_path_pic1 = 'api_usage/temp_pic/pic1.jpg'
    image_path_pic2 = 'api_usage/temp_pic/pic2.jpg'

    image1 = cv2.imread(image_path_pic1, cv2.IMREAD_COLOR)
    image2 = cv2.imread(image_path_pic2, cv2.IMREAD_COLOR)
    '''
    cv2.imshow("pic1", image1)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    cv2.imshow("pic2", image2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)

    try:
        dets_pic1 = faceDetModelHandler.inference_on_image(image1)
        dets_pic2 = faceDetModelHandler.inference_on_image(image2)

    except Exception as e:
        logger.error('Face detection failed!')
        logger.error(e)
        sys.exit(-1)


    pic1_box = dets_pic1

    line1 = str(int(pic1_box[0][0])) + " " + str(int(pic1_box[0][1])) + " " + \
            str(int(pic1_box[0][2])) + " " + str(int(pic1_box[0][3])) + " " + \
            str(pic1_box[0][4]) + " \n"


    pic2_box = dets_pic2
    line2 = str(int(pic2_box[0][0])) + " " + str(int(pic2_box[0][1])) + " " + \
            str(int(pic2_box[0][2])) + " " + str(int(pic2_box[0][3])) + " " + \
            str(pic2_box[0][4]) + " \n"
#★☆★★☆★★☆★★☆★detect fin★☆★★☆★★☆★★☆★★☆★
#-------------------alignment start-------------------------
#change model_category
    model_category = 'face_alignment'
    model_name = model_conf[scene][model_category]

    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)


    try:
        model, cfg = faceAlignModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)


    faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)


    image_path_pic1 = 'api_usage/temp_pic/pic1.jpg'
    image_det_txt_path_pic1 = 'api_usage/temp_pic/pic1_detect_res.txt'
    image_pic1 = cv2.imread(image_path_pic1, cv2.IMREAD_COLOR)

    image_path_pic2 = 'api_usage/temp_pic/pic2.jpg'
    image_det_txt_path_pic2 = 'api_usage/temp_pic/pic2_detect_res.txt'
    image_pic2 = cv2.imread(image_path_pic2, cv2.IMREAD_COLOR)

# ----------------------pic landmark----------------------
    try:
        line1 = line1.strip().split()
        det_pic1 = np.asarray(list(map(int, line1[0:4])), dtype=np.int32)
        landmarks_pic1 = faceAlignModelHandler.inference_on_image(image1, det_pic1)
        line2 = line2.strip().split()
        det_pic2 = np.asarray(list(map(int, line2[0:4])), dtype=np.int32)
        landmarks_pic2 = faceAlignModelHandler.inference_on_image(image2, det_pic2)

    except Exception as e:
        logger.error('Face landmark failed!')
        logger.error(e)
        sys.exit(-1)
# ★☆★★☆★★☆★★☆★alignment fin★☆★★☆★★☆★★☆★★☆★
#-----------------------crop start--------------------------
    face_cropper = FaceRecImageCropper()

    image1 = cv2.imread(image_path_pic1)
    image2 = cv2.imread(image_path_pic2)
    #landmark의 차원수를 낮추기 위해 flatten() 사용
    flatten_landmarks1 = np.array(landmarks_pic1).flatten().tolist()
    flatten_landmarks2 = np.array(landmarks_pic2).flatten().tolist()

    landmarks1 =  [float(num) for num in flatten_landmarks1]
    cropped_image1 = face_cropper.crop_image_by_mat(image1, landmarks1)

    landmarks2 = [float(num) for num in flatten_landmarks2]
    cropped_image2 = face_cropper.crop_image_by_mat(image2, landmarks2)
    '''
    cv2.imshow("cropped_image1", cropped_image1)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imshow("cropped_image2", cropped_image2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    #크롭된 이미지 저장하는 문장
    #cv2.imwrite('api_usage/temp_pic/pic1_cropped.jpg', cropped_image1)
    #cv2.imwrite('api_usage/temp_pic/pic2_cropped.jpg', cropped_image2)
# ★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆crop fin★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆★☆
#----------------------feature start--------------------------
    model_category = 'face_recognition'
    model_name = model_conf[scene][model_category]
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)

    try:
        model, cfg = faceRecModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)

    faceRecModelHandler = FaceRecModelHandler(model, 'cpu', cfg)

    try:  # 여기서 오류
        #크롭된 이미지의 피쳐를 뽑아냄
        feature1 = faceRecModelHandler.inference_on_image(cropped_image1)
        feature2 = faceRecModelHandler.inference_on_image(cropped_image2)

    except Exception as e:
        logger.error('Failed to extract facial features!')
        logger.error(e)
        sys.exit(-1)

# ★☆★★☆★★☆★★☆★feature fin★☆★★☆★★☆★★☆★★☆★
#----------------------pipline start--------------------------
#점수 계산 pic1_crop의 feature와 pic2_crop의 feature dot 연산
    score = np.dot(feature1, feature2)
    print('The score for pic1 and pic2 is', score)
