"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 
"""
import sys
sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')
import torch
import yaml
import cv2
import numpy as np
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.Loader)

if __name__ == '__main__':
    # common setting for all model, need not modify.
    model_path = 'models'

    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]

    logger.info('Start to load the face landmark model...')
    # load model
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')

    try:
        model, cfg = faceAlignModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face landmark model!')

    faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)

    # read image
    #test1
    image_path = 'api_usage/test_images/test1.jpg'
    image_det_txt_path = 'api_usage/test_images/test1_detect_res.txt'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #test2
    image_path2 = 'api_usage/test_images/test2.jpg'
    image_det_txt_path2 = 'api_usage/test_images/test2_detect_res.txt'
    image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

#--------------------------------------------------------------------------------------
#test1
    with open(image_det_txt_path, 'r') as f:
        lines = f.readlines()
    try:
        for i, line in enumerate(lines):
            line = line.strip().split()
            det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
            landmarks = faceAlignModelHandler.inference_on_image(image, det)

            save_path_img = 'api_usage/temp/test1_' + 'landmark_res' + str(i) + '.jpg'
            save_path_txt = 'api_usage/temp/test1_' + 'landmark_res' + str(i) + '.txt'
            image_show = image.copy()
            with open(save_path_txt, "w") as fd:
                for (x, y) in landmarks.astype(np.int32):
                    cv2.circle(image_show, (x, y), 2, (255, 0, 0),-1)
                    line = str(x) + ' ' + str(y) + ' '
                    fd.write(line)
            cv2.imwrite(save_path_img, image_show)
    except Exception as e:
        logger.error('Face landmark failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successful face landmark!')

if __name__ == 'face_alignment':
    # common setting for all model, need not modify.
    model_path = 'models'

    # model setting, modified along with model
    scene = 'non-mask'
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]

    logger.info('Start to load the face landmark model...')
    # load model
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error('Failed to parse model configuration file!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully parsed the model configuration file model_meta.json!')

    try:
        model, cfg = faceAlignModelLoader.load_model()
    except Exception as e:
        logger.error('Model loading failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successfully loaded the face landmark model!')

    faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)

    # read image
    #test1
    image_path = 'api_usage/hallway/merged.jpg'
    image_det_txt_path = 'api_usage/hallway/test1_detect_res.txt'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#--------------------------------------------------------------------------------------
#test1
    with open(image_det_txt_path, 'r') as f:
        lines = f.readlines()
        print(lines)
    try:
        for i, line in enumerate(lines):
            line = line.strip().split()
            print(line)
            det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
            landmarks = faceAlignModelHandler.inference_on_image(image, det)

            save_path_img = 'api_usage/hallway/test1_' + 'landmark_res' + str(i) + '.jpg'
            save_path_txt = 'api_usage/hallway/test1_' + 'landmark_res' + str(i) + '.txt'
            image_show = image.copy()
            with open(save_path_txt, "w") as fd:
                for (x, y) in landmarks.astype(np.int32):
                    cv2.circle(image_show, (x, y), 2, (255, 0, 0),-1)
                    line = str(x) + ' ' + str(y) + ' '
                    fd.write(line)
            cv2.imwrite(save_path_img, image_show)
    except Exception as e:
        logger.error('Face landmark failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successful face landmark!')

#--------------------------------------------------------------------------------------
"""#test2
    with open(image_det_txt_path2, 'r') as f:
        lines = f.readlines()
    try:
        for i, line in enumerate(lines):
            line = line.strip().split()
            det2 = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
            landmarks = faceAlignModelHandler.inference_on_image(image2, det2)

            save_path_img2 = 'api_usage/temp/test1_' + 'landmark_res' + '1'+ '.jpg'
            save_path_txt2 = 'api_usage/temp/test1_' + 'landmark_res' + '1' + '.txt'
            image_show2 = image2.copy()
            with open(save_path_txt2, "w") as fd:
                for (x, y) in landmarks.astype(np.int32):
                    cv2.circle(image_show2, (x, y), 2, (255, 0, 0),-1)
                    line = str(x) + ' ' + str(y) + ' '
                    fd.write(line)
            cv2.imwrite(save_path_img2, image_show2)
    except Exception as e:
        logger.error('Face landmark failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Successful face landmark!')
"""