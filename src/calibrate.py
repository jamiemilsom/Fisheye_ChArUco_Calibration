import os
import numpy as np
import cv2
import cv2.aruco
import json


class CameraCalibrator:
    def __init__(self, aruco_dict, squares_vertically, squares_horizontally, square_length, marker_length):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.squares_vertically = squares_vertically
        self.squares_horizontally = squares_horizontally
        self.square_length = square_length
        self.marker_length = marker_length
        
        self.board = cv2.aruco.CharucoBoard((self.squares_vertically, self.squares_horizontally), self.square_length, self.marker_length, self.aruco_dict)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        
    def make_charuco_board(self):
        """
        This function creates a ChArUco board and saves it to a file.
        """
        size_ratio = self.squares_horizontally / self.squares_vertically
        img = cv2.aruco.CharucoBoard.generateImage(self.board, (640, int(640*size_ratio)), marginSize=20)
        cv2.imshow("charuco_board.png", img)
        cv2.imwrite("charuco_board.png", img)
        

    def visualise_aruco_markers(self, image_paths, graysale=True,refine = True, window_size=(1080,720)):
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
                
                # print(f"Found {len(marker_ids)} markers in {image_path}")
                total_detected_markers += len(marker_ids)
                detected_markers_list.append(len(marker_ids))
                
                (h, w) = image.shape[:2]
                new_width = window_size[0]
                new_height = int(h * (new_width / w))
                resized_image = cv2.resize(image, (new_width, new_height))
                
                scale_factor = new_width / w
                
                for i, corner in enumerate(marker_corners):
 
                    corner_resized = corner[0] * scale_factor
                    corner_resized = corner_resized.astype(int)
                    
                    cv2.polylines(resized_image, [corner_resized], True, (0, 255, 0), 2)
                    center = np.mean(corner_resized, axis=0).astype(int)
                    cv2.putText(resized_image, str(marker_ids[i][0]), tuple(center), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
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

        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, self.board, image.shape[:2], None, None)
        print('Calibration complete.')
        
        print('Camera matrix: ', camera_matrix)
        print('Distortion coefficients: ', dist_coeffs)
        np.save('camera_matrix.npy', camera_matrix)
        np.save('dist_coeffs.npy', dist_coeffs)
        
        
    def fisheye_calibration(self, image_paths, output_path='fisheye_calibration.json'):
        params = cv2.aruco.DetectorParameters()
        image_paths.sort()

        
    def undistort_images(self, image_paths, output_path):
        """
        This function undistorts images using the camera matrix and distortion coefficients.
        Args:
        image_paths: A list of file paths to images.
        output_path: A string representing the output directory.
        """
        camera_matrix = np.load('camera_matrix.npy')
        dist_coeffs = np.load('dist_coeffs.npy')

        image_paths.sort()

        os.makedirs(output_path, exist_ok=True)  # Create directory if it doesn't exist

        for image_file in image_paths:
            image = cv2.imread(image_file)
            undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

            # Extract filename without extension
            filename, _ = os.path.splitext(os.path.basename(image_file))
            output_filename = os.path.join(output_path, f"{filename}_undistorted.jpg")

            # Save the undistorted image
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
instaCam.visualise_aruco_markers(file_paths,graysale=True,refine=True)
instaCam.pinhole_calibration(file_paths)
instaCam.fisheye_calibration(file_paths)
instaCam.undistort_images(file_paths, OUTPUT_PATH)
