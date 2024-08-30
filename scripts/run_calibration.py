import os
import sys
import numpy as np
import cv2
import cv2.aruco
import json
from typing import Generator, Tuple, Optional
from calibration import FisheyeCalibrator, PinholeCalibrator, OmnidirectionalCalibrator
import matplotlib.pyplot as plt

ARUCO_DICT = cv2.aruco.DICT_5X5_100
SQUARES_VERTICALLY = 12
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.06
MARKER_LENGTH = 0.045
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

instaCamFisheye = FisheyeCalibrator(
    ARUCO_DICT, SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, SQUARE_LENGTH, MARKER_LENGTH, 
    calibration_images_dir= os.path.join(CURRENT_PATH,'../data/calibration/images/'),
    raw_images_dir= os.path.join(CURRENT_PATH,'../data/raw_images/descent_1')
    )

instaCamPinhole = PinholeCalibrator(
    ARUCO_DICT, SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, SQUARE_LENGTH, MARKER_LENGTH, 
    calibration_images_dir= os.path.join(CURRENT_PATH,'../data/calibration/images/'),
    raw_images_dir= os.path.join(CURRENT_PATH,'../data/raw_images/descent_1')
    )

instaCamOmnidirectional = OmnidirectionalCalibrator(
    ARUCO_DICT, SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, SQUARE_LENGTH, MARKER_LENGTH, 
    calibration_images_dir= os.path.join(CURRENT_PATH,'../data/calibration/images/'),
    raw_images_dir= os.path.join(CURRENT_PATH,'../data/raw_images/descent_1')
    )

instaCamFisheye.calibrate()
# instaCamPinhole.calibrate()

# image = cv2.imread('/home/jamie/Documents/reconstruction/data/calibration/resized_images/IMG_20231011_152834_00_068.jpg')
# instaCamPinhole.undistort_image(image=image,show_image=True,save_image=False,balance=0.0)



    
    
# instaCamOmnidirectional.calibrate()
# image = cv2.imread('/home/jamie/Documents/reconstruction/data/calibration/images/IMG_20231011_151416_00_001.jpg')
# instaCamOmnidirectional.undistort_image(image=image,show_image=True)

instaCamFisheye.export_camera_params_colmap()
# instaCamPinhole.export_camera_params_colmap()