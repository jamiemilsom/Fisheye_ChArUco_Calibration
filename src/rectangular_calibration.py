import os
import numpy as np
import cv2
import cv2.aruco
import json
from virtual_camera import RectangularCamera
from calibrate import CameraCalibrator

ARUCO_DICT = cv2.aruco.DICT_5X5_100
SQUARES_VERTICALLY = 12
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.06
MARKER_LENGTH = 0.045

input_folder = '/home/jamie/Documents/reconstruction/data/calibration/jpg'
rectangular_output_folder = '/home/jamie/Documents/reconstruction/data/calibration/virtual_cameras/rectangular'



rectangular_cam = RectangularCamera(input_folder, rectangular_output_folder, sections=5, overlap_ratio=0.25)

for image_path in rectangular_cam.input_image_list:
    print('Splitting image: ', image_path)
    rectangular_cam.split_image(image_path)
    
instaCam = CameraCalibrator(
    ARUCO_DICT, SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, SQUARE_LENGTH, MARKER_LENGTH
    )

def image_list(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]

folder_names = [f for f in os.listdir(rectangular_output_folder) if os.path.isdir(os.path.join(rectangular_output_folder, f))]
print('Folder names:', folder_names)

for folder_name in folder_names:
    print('Calibrating camera ', folder_name)
    instaCam.calibrate(image_paths=image_list(os.path.join(rectangular_output_folder, folder_name)), model='fisheye', output_path='rectangular_camera_' + folder_name + '.json')

