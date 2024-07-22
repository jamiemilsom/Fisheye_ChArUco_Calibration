import os
import numpy as np
import cv2
import cv2.aruco
import json
import matplotlib.pyplot as plt
import time


class CameraCalibrator:
    def __init__(self, aruco_dict, squares_vertically, squares_horizontally, square_length, marker_length):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.squares_vertically = squares_vertically
        self.squares_horizontally = squares_horizontally
        self.square_length = square_length
        self.marker_length = marker_length
        
        self.board = cv2.aruco.CharucoBoard((self.squares_vertically, self.squares_horizontally), self.square_length, self.marker_length, self.aruco_dict)
        self.K = None
        self.D = None
        
        
    def make_charuco_board(self):
        """
        This function creates a ChArUco board and saves it to a file.
        """
        size_ratio = self.squares_horizontally / self.squares_vertically
        img = cv2.aruco.CharucoBoard.generateImage(self.board, (640, int(640*size_ratio)), marginSize=20)
        cv2.imshow("charuco_board.png", img)
        cv2.imwrite("charuco_board.png", img)
        

    def visualise_aruco_markers(self, image_paths, graysale=True,refine = True,refine_with_charuco=True, window_size=(1080,720)):
        """
        This function iterates through image files, detects ArUco markers,
        and displays the image with markers highlighted at a reduced size.
        Args:
        image_paths: A list of file paths to images.
        window_size (optional): A tuple of integers to set the window size.
        """
        image_paths.sort()
        params = cv2.aruco.DetectorParameters()
        total_detected_markers = 0
        detected_markers_list = []
        total_images = len(image_paths)
        
        
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if graysale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)             
                
            marker_corners, marker_ids, rejected_corners = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=params)
            
            if marker_ids is not None and len(marker_ids) > 0:
                if refine:
                    refined_corners, refined_ids, _, _ = cv2.aruco.refineDetectedMarkers(
                    image, self.board, marker_corners, marker_ids, rejected_corners,
                    parameters=params
                )
                    marker_corners = refined_corners
                    marker_ids = refined_ids
                    cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
                    
                if refine_with_charuco:
                    charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, self.board)
                    if charuco_retval:
                        cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
                        if len(charuco_corners) > len(marker_corners):
                            print(f"Refining with ChArUco: {len(charuco_corners)} Refining with MarkerIds: {len(marker_corners)} for {image_path}")
                
                # print(f"Found {len(marker_ids)} markers in {image_path}")
                total_detected_markers += len(marker_ids)
                detected_markers_list.append(len(marker_ids))
                
                (h, w) = image.shape[:2]
                new_width = window_size[0]
                new_height = int(h * (new_width / w))
                resized_image = cv2.resize(image, (new_width, new_height))
                
                scale_factor = new_width / w
                
                # for i, corner in enumerate(marker_corners):
 
                #     corner_resized = corner[0] * scale_factor
                #     corner_resized = corner_resized.astype(int)
                    
                #     cv2.polylines(resized_image, [corner_resized], True, (0, 255, 0), 2)
                #     center = np.mean(corner_resized, axis=0).astype(int)
                #     cv2.putText(resized_image, str(marker_ids[i][0]), tuple(center), 
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                cv2.imshow('Detected ArUco Markers', resized_image)
                cv2.waitKey(1)

        cv2.destroyAllWindows()
        print(f"Total number of images: {total_images}")
        print(f"Average number of markers per image: {total_detected_markers / total_images:.2f}")
        print(f"Detected markers per image: {detected_markers_list}")
        
        
    def pinhole_calibration(self, image_paths, output_path='calibration_output.json'):

        params = cv2.aruco.DetectorParameters()
        
        image_paths.sort()
        
        all_charuco_corners = []
        all_charuco_ids = []

        for image_file in image_paths:
            image = cv2.imread(image_file)
            image_copy = image.copy()
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=params)
            
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
                charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, self.board)
                if charuco_retval:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                else:
                    print(f"Could not interpolate corners for {image_file}")
                    
        print('Starting calibration...')

        retval, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, self.board, image.shape[:2], None, None)

        
        print('K: ', K)
        print('Distortion coefficients: ', D)
        np.save('K.npy', K)
        np.save('D.npy', D)
        
                
    def fisheye_calibration(self, image_paths, output_path='fisheye_calibration.json'):
        params = cv2.aruco.DetectorParameters()
        
        image_paths.sort()
        
        all_charuco_corners = []
        all_charuco_ids = []
        object_points = []
        image_points = []
        
        print('Detecting ArUco markers and interpolating ChArUco corners...')

        for image_file in image_paths:
            image = cv2.imread(image_file)
            image_copy = image.copy()
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=params)
            
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
                charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, self.board)
                if charuco_retval:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                else:
                    print(f"Could not interpolate corners for {image_file}")
        
        print('Mapping ChArUco corners to 3D space...')
        objPoints = self.board.getChessboardCorners()

        
        for corners, ids in zip(all_charuco_corners, all_charuco_ids): 
            obj_points = objPoints[ids]
            object_points.append(obj_points)
            image_points.append(corners)
            
        print('Starting calibration...')
        
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        
        retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(object_points,image_points,image.shape[:2],K,D)

        
        print('K: ', K)
        print('Distortion coefficients: ', D)
        np.save('K.npy', K)
        np.save('D.npy', D)
        
        
    def undistort_images(self, image_paths, output_path, undistort_type = 1, scale = 0.7):
        """
        This function undistorts images using the camera matrix and distortion coefficients.
        Args:
        image_paths: A list of file paths to images.
        output_path: A string representing the output directory.
        undistort_type: A int to specify the type of undistortion (0: Pinhole, 1: Fisheye via RectifyMap, 2: Fisheye via Undistort)
        scale: A float for the output image size
        """
        undist_types = ['Pinhole', 'Fisheye_RectifyMap', 'Fisheye_Undistort']
        
        K = np.load('K.npy')
        D = np.load('D.npy')

        image_paths.sort()

        os.makedirs(output_path, exist_ok=True)  # Create directory if it doesn't exist

        for image_file in image_paths:
            image = cv2.imread(image_file)
            h, w = image.shape[:2]
            
            if undistort_type == 2:
                undistorted_image = cv2.fisheye.undistortImage(image, K, D, Knew=K)
            
            if undistort_type == 1:
                
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, (w, h), np.eye(3), balance=scale
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
                )
                undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            
            if undistort_type == 0:
            
                new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), scale, (w, h))
                undistorted_image = cv2.undistort(image, K, D, None, new_K)

            filename, _ = os.path.splitext(os.path.basename(image_file))
            output_filename = os.path.join(output_path, f"{filename}_undistorted_{undist_types[undistort_type]}.jpg")


            cv2.imwrite(output_filename, undistorted_image)

        cv2.destroyAllWindows()
    
            
        


ARUCO_DICT = cv2.aruco.DICT_5X5_100
SQUARES_VERTICALLY = 12
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.06
MARKER_LENGTH = 0.045
PATH_TO_YOUR_IMAGES = '/home/jamie/Documents/reconstruction/data/calibration/jpg'
OUTPUT_PATH = '/home/jamie/Documents/reconstruction/data/calibration/undistorted'

file_paths = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".jpg")]

instaCam = CameraCalibrator(ARUCO_DICT, SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, SQUARE_LENGTH, MARKER_LENGTH)
# instaCam.visualise_aruco_markers(file_paths,graysale=True,refine=True,refine_with_charuco=False, window_size=(1080,720))
instaCam.pinhole_calibration(file_paths)
instaCam.undistort_images(file_paths, OUTPUT_PATH,undistort_type=0, scale=0)
instaCam.fisheye_calibration(file_paths)
instaCam.undistort_images(file_paths, OUTPUT_PATH,undistort_type=1, scale=0)
instaCam.undistort_images(file_paths, OUTPUT_PATH,undistort_type=2, scale=0)
