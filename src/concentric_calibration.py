import os
import numpy as np
import cv2
import cv2.aruco
import json
from virtual_camera import ConcentricCamera
from calibrate import CameraCalibrator

ARUCO_DICT = cv2.aruco.DICT_5X5_100
SQUARES_VERTICALLY = 12
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.06
MARKER_LENGTH = 0.045

input_folder = '/home/jamie/Documents/reconstruction/data/calibration/jpg'
concentric_output_folder = '/home/jamie/Documents/reconstruction/data/calibration/virtual_cameras/concentric'
camera_0_image_folder = '/home/jamie/Documents/reconstruction/data/calibration/virtual_cameras/concentric/camera_0'
camera_1_image_folder = '/home/jamie/Documents/reconstruction/data/calibration/virtual_cameras/concentric/camera_1'


concentric_cam = ConcentricCamera(input_folder, concentric_output_folder, splits=[0.6], overlap_ratio=0.1)

for image_path in concentric_cam.input_image_list:
    print('Splitting image: ', image_path)
    concentric_cam.split_image(image_path)
    
instaCam = CameraCalibrator(
    ARUCO_DICT, SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, SQUARE_LENGTH, MARKER_LENGTH
    )

def image_list(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]

print('Calibrating camera 0')
instaCam.calibrate(image_paths=image_list(camera_0_image_folder), model='fisheye', output_path='concentric_camera_0.json')
print('Calibrating camera 1')
instaCam.calibrate(image_paths=image_list(camera_1_image_folder), model='fisheye', output_path='concentric_camera_1.json')
