import os
import numpy as np
import cv2
from virtual_camera import RectangularCamera, ConcentricCamera


ARUCO_DICT = cv2.aruco.DICT_5X5_100
SQUARES_VERTICALLY = 12
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.06
MARKER_LENGTH = 0.045

input_folder = os.path.join(os.path.dirname(__file__), "../data/raw_images/")
print(input_folder)
rectangular_output_folder = os.path.join(os.path.dirname(__file__), "../data/virtual_cameras/rectangular_images/")
concentric_output_folder = os.path.join(os.path.dirname(__file__), "../data/virtual_cameras/concentric_images/")

rectangular_cam = RectangularCamera(input_folder, rectangular_output_folder, sections=9, overlap_ratio=0.1)
concentric_cam = ConcentricCamera(input_folder, concentric_output_folder, splits=[0.6], overlap_ratio=0.1)


for image_path in concentric_cam.input_image_list:
    concentric_cam.split_image(image_path)

for image_path in rectangular_cam.input_image_list:
    rectangular_cam.split_image(image_path)
