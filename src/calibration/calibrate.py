import os
import numpy as np
import cv2
import cv2.aruco
import json
from typing import Generator, Tuple, Optional


class CharucoCalibrator:
    def __init__(self, aruco_dict: int, squares_vertically: int, squares_horizontally: int, square_length: float, marker_length:float, calibration_images_dir:str = None, raw_images_dir:str = None):
        """
        Initializes the CharucoCalibrator with the specified parameters.

        Args:
            aruco_dict (int): The ArUco dictionary to use.
            squares_vertically (int): Number of squares vertically in the ChArUco board.
            squares_horizontally (int): Number of squares horizontally in the ChArUco board.
            square_length (float): Length of each square in the ChArUco board.
            marker_length (float): Length of each ArUco marker in the ChArUco board.
            calibration_images_dir (str, optional): Directory containing calibration images.
            raw_images_dir (str, optional): Directory containing raw images to be processed.
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.squares_vertically = squares_vertically
        self.squares_horizontally = squares_horizontally
        self.square_length = square_length
        self.marker_length = marker_length
        self.calibration_image_dir = calibration_images_dir
        self.raw_image_dir = raw_images_dir
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.board = cv2.aruco.CharucoBoard((self.squares_vertically, self.squares_horizontally), self.square_length, self.marker_length, self.aruco_dict)
        self.params = cv2.aruco.DetectorParameters()
        self.K = np.zeros((3, 3))
        self.D = None
        
    def image_generator(self, folder_path) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Generates NumPy arrays of images from the specified directory in sorted order.

        Args:
            folder_path (str): The path to the directory containing images.

        Yields:
            Tuple[str, np.ndarray]: A tuple containing the filename and the image as a NumPy array.
        """
        image_paths = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        for img_path in image_paths:
            try:
                yield os.path.basename(img_path), cv2.imread(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")


    def calibration_images(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Generator function that yields images from the calibration images directory.

        Yields:
            Tuple[str, np.ndarray]: A tuple containing the filename and the image as a NumPy array.
        """
        yield from self.image_generator(self.calibration_image_dir)

    def raw_images(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Generator function that yields images from the raw images directory.

        Yields:
            Tuple[str, np.ndarray]: A tuple containing the filename and the image as a NumPy array.
        """
        yield from self.image_generator(self.raw_image_dir)

    
    def generate_blank_board(self) -> None:
        """
        Creates a blank black board and saves it as data/charuco_board.png.
        """
        size_ratio = self.squares_horizontally / self.squares_vertically
        img_width = 640
        img_height = int(640 * size_ratio)

        blank_board = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/calibration/charuco_board.png')
        cv2.imwrite(output_path, blank_board)
        print(f"Blank board saved to {output_path}")
        

        
    def generate_charuco_board(self) -> None:
        """
        This function creates a ChArUco board and saves it data/charuco_board.png.
        """
        size_ratio = self.squares_horizontally / self.squares_vertically
        img = cv2.aruco.CharucoBoard.generateImage(self.board, (640, int(640*size_ratio)), marginSize=20)
        cv2.imshow("charuco_board.png", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/calibration/charuco_board.png')
        cv2.imwrite(output_path, img)
        print(f"ChArUco board saved to {output_path}")
        
        

    def detect_aruco_markers(self, image: np.ndarray,image_name: str = None, graysale=True,verbose = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detects ArUco markers in an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            image_name (str, optional): Name of the image for logging purposes.
            graysale (bool): Whether to convert the image to grayscale. Defaults to True.
            verbose (bool): Whether to print detection results. Defaults to True.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: A tuple of (ids, corners) if markers are detected, otherwise None.
        """
        
        if graysale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)             
        
        marker_corners, marker_ids, rejected_corners = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters=self.params)
        
        if marker_ids is not None and len(marker_ids) > 0:
            refined_corners, refined_ids, _, _ = cv2.aruco.refineDetectedMarkers(
                image, self.board, marker_corners, marker_ids, rejected_corners,
                parameters=self.params
            )
            marker_corners = refined_corners
            marker_ids = refined_ids
            if verbose:
                print(f"{len(marker_ids)} ArUco markers detected in {image_name}")
            return marker_ids, marker_corners
            
            
        else:
            if verbose:
                print(f"No ArUco markers detected in {image_name}")
            return None, None
    


    def detect_charuco_corners(self, image: np.ndarray,image_name: str = None, grayscale=True,verbose = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Detects ChArUco corners in an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            image_name (str, optional): Name of the image for logging purposes.
            grayscale (bool): Whether to convert the image to grayscale. Defaults to True.
            verbose (bool): Whether to print detection results. Defaults to True.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: A tuple of (charuco_ids, charuco_corners) if corners are detected, otherwise None.
        """

        marker_ids, marker_corners = self.detect_aruco_markers(image, grayscale,verbose=verbose)
        
        if marker_ids is not None and len(marker_ids) > 0:

            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, self.board)
                
            if charuco_retval:                
                num_charuco_corners = len(charuco_corners)
            else:
                num_charuco_corners = 0
            if verbose:
                print(f"Detected {num_charuco_corners} charuco corners in {image_name}")
            return charuco_ids, charuco_corners
        else:
            if verbose:
                print(f"No charuco corners detected in {image_name}")
            return None, None
        
    def show_aruco_markers(self, image: np.ndarray,image_name:str = None, window_size = (480,480),verbose = True) -> None:
        """
        Displays the detected ArUco markers in an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            image_name (str, optional): Name of the image for logging purposes.
            window_size (tuple): A tuple of (width, height) for the window size. Defaults to (480, 480).
            verbose (bool): Whether to print detection results. Defaults to True.
        """


        image_copy = image.copy()
        marker_ids, marker_corners = self.detect_aruco_markers(image_copy,image_name,verbose=verbose)
        cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
        cv2.imshow("Detected ArUco Markers", cv2.resize(image_copy, window_size))
        if verbose:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        
    def show_charuco_corners(self, image: np.ndarray, image_name:str = None, window_size = (480,480),verbose = True) -> None:
        """
        Displays the detected ChArUco corners in an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            image_name (str, optional): Name of the image for logging purposes.
            window_size (tuple): A tuple of (width, height) for the window size. Defaults to (480, 480).
            verbose (bool): Whether to print detection results. Defaults to True.
        """

        image_copy = image.copy()
        charuco_ids, charuco_corners = self.detect_charuco_corners(image_copy, image_name,verbose=verbose)
        cv2.aruco.drawDetectedCornersCharuco(image_copy, charuco_corners, charuco_ids)
        cv2.imshow("Detected Charuco Corners", cv2.resize(image_copy, window_size))
        if verbose:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        
    def save_camera_parameters(self, file_path):
        """
        Saves camera intrinsic parameters to a JSON file.

        Args:
            file_path (str): Path to save the camera parameters.
        """

        json_data = {
            'K': self.K.tolist(),
            'D': self.D.tolist()
        }

        json.dump(json_data, open(file_path, 'w'))    
        
        
    def load_camera_parameters(self, file_path: str = None) -> None:
        """
        Loads camera intrinsic parameters from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing camera parameters.

        Returns:
            tuple: A tuple containing the camera matrix (K) and distortion coefficients (D).
        """
        with open(file_path, 'r') as f:
            camera_intrinsics = json.load(f)

        self.K = np.array(camera_intrinsics['K'])
        self.D = np.array(camera_intrinsics['D'])
        
        
        
        
class FisheyeCalibrator(CharucoCalibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def calibrate(self, grayscale = True, calibration_filename ='fisheye_calibration.json', window_size=(480,480),verbose = False) -> None:
        """
        Performs Fisheye camera calibration using ChArUco markers.

        Args:
            grayscale (bool): Whether to convert images to grayscale. Defaults to True.
            calibration_filename (str): Filename to save the calibration results. Defaults to 'fisheye_calibration.json'.
            window_size (tuple): Size of the window for displaying images. Defaults to (480, 480).
            verbose (bool): Whether to print detailed information during calibration. Defaults to False.
        """
        
        all_charuco_corners = []
        all_charuco_ids = []

        
        for image_name, image in self.calibration_images():
            charuco_ids, charuco_corners = self.detect_charuco_corners(image=image, image_name=image_name,grayscale=grayscale,verbose=verbose)
            if charuco_ids is not None and len(charuco_ids) > 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
            else:
                print(f"Could not interpolate corners for {image_name}")
                self.show_aruco_markers(image,image_name = image_name, window_size=window_size,verbose=verbose)
        
        if not all_charuco_corners:
            print("No charuco corners detected in any images.")
            return
        
        object_points = []
        image_points = []
        
        objPoints = self.board.getChessboardCorners()

        count = 0
        for corners, ids in zip(all_charuco_corners, all_charuco_ids): 
            if len(corners) >= 4:
                count += 1
                obj_points = objPoints[ids]
                object_points.append(obj_points)
                image_points.append(corners)
        
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))
        
        print('Number of images used for calibration: ', count)
        
        retval, self.K, self.D, rvecs, tvecs = cv2.fisheye.calibrate(object_points,image_points,image.shape[:2],self.K,self.D)

        print('K: ', self.K)
        print('Distortion coefficients: ', self.D)
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data/calibration/camera_intrinsics/{calibration_filename}')
        
        self.save_camera_parameters(output_path)
        print(f"Camera calibration data saved to {output_path}")
        
        
    def undistort_image(self, image: np.ndarray, image_name:str = None, calibration_filename = 'fisheye_calibration.json', balance = 1, show_image = True, save_image = True, output_path: str = None,window_size = (480,480)) -> np.ndarray:
        """
        Undistorts a fisheye image using the calibrated parameters.

        Args:
            image (np.ndarray): The input image to undistort.
            image_name (str, optional): Name of the image for saving purposes.
            calibration_filename (str): Filename of the calibration file. Defaults to 'fisheye_calibration.json'.
            balance (float): Balance parameter for undistortion. Defaults to 1.
            show_image (bool): Whether to display the undistorted image. Defaults to True.
            save_image (bool): Whether to save the undistorted image. Defaults to True.
            output_path (str, optional): Path to save the undistorted image.
            window_size (tuple): Size of the window for displaying images. Defaults to (480, 480).

        Returns:
            np.ndarray: The undistorted image.
        """
        try:

            calibration_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data/calibration/camera_intrinsics/{calibration_filename}') 
            self.load_camera_parameters(calibration_path)
                    
            h, w = image.shape[:2]

            new_K = self.K.copy()
            new_K[0,0] *= balance  # Scale fx
            new_K[1,1] *= balance  # Scale fy
            
            undistorted_image = cv2.fisheye.undistortImage(image, self.K, self.D, Knew=new_K,new_size=(w,h))
            
            if show_image:
                cv2.imshow("Undistorted Image", cv2.resize(undistorted_image, window_size))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            if save_image:
                if output_path is None:
                    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data/undistorted_images/{image_name}')
                cv2.imwrite(output_path, undistorted_image)
                print(f"Undistorted image saved to {output_path}")
            
            return undistorted_image
        except Exception as e:
            print(f"Error undistorting image {image_name}: {e}")
            return None
        
    def export_camera_params_colmap(self, calibration_path: str = None) -> None:
        """
        Exports camera parameters in COLMAP format.
        """
        if calibration_path is None:
            calibration_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data/calibration/camera_intrinsics/fisheye_calibration.json')
        
        self.load_camera_parameters(calibration_path)
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        k1 = self.D[0][0]
        k2 = self.D[1][0]
        k3 = self.D[2][0]
        k4 = self.D[3][0]
        print('Camera parameters for COLMAP:')
        print(f"OPENCV_FISHEYE: {fx}, {fy}, {cx}, {cy}, {k1}, {k2}, {k3}, {k4}")
        
        
class PinholeCalibrator(CharucoCalibrator):
        
        
    def calibrate(self, grayscale = True, calibration_filename ='pinhole_calibration.json', window_size=(480,480),verbose = False) -> None:
        """
        Performs pinhole camera calibration using ChArUco markers.

        Args:
            grayscale (bool): Whether to convert images to grayscale. Defaults to True.
            calibration_filename (str): Filename to save the calibration results. Defaults to 'pinhole_calibration.json'.
            window_size (tuple): Size of the window for displaying images. Defaults to (480, 480).
            verbose (bool): Whether to print detailed information during calibration. Defaults to False.
        """

    
        
        all_charuco_corners = []
        all_charuco_ids = []
        
        for image_name, image in self.calibration_images():
            charuco_ids, charuco_corners = self.detect_charuco_corners(image=image, image_name=image_name,grayscale=grayscale,verbose=verbose)
            if charuco_ids is not None and len(charuco_ids) > 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
            else:
                print(f"Could not interpolate corners for {image_name}")
                self.show_aruco_markers(image,image_name = image_name, window_size=window_size,verbose=verbose)
                
        if not all_charuco_corners:
            print("No charuco corners detected in any images.")
            return
                    
        
        self.K = np.zeros((3, 3))
        self.D = np.zeros((5, 1))
        retval, self.K, self.D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, self.board, image.shape[:2], None, None)
                    
        print('K: ', self.K)
        print('Distortion coefficients: ', self.D)
        
        print('Number of images used for calibration: ', len(all_charuco_corners))
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data/calibration/camera_intrinsics/{calibration_filename}')
        self.save_camera_parameters(output_path)
        print(f"Camera calibration data saved to {output_path}")  
        
        
    def undistort_image(self, image: np.ndarray, image_name:str = None, calibration_filename = 'pinhole_calibration.json', balance = 1, show_image = True, save_image = True, output_path: str = None,window_size = (480,480)) -> np.ndarray:
        """
        Undistorts a pinhole camera image using the calibrated parameters.

        Args:
            image (np.ndarray): The input image to undistort.
            image_name (str, optional): Name of the image for saving purposes.
            calibration_filename (str): Filename of the calibration file. Defaults to 'pinhole_calibration.json'.
            balance (float): Balance parameter for undistortion. Defaults to 1.
            show_image (bool): Whether to display the undistorted image. Defaults to True.
            save_image (bool): Whether to save the undistorted image. Defaults to True.
            output_path (str, optional): Path to save the undistorted image.
            window_size (tuple): Size of the window for displaying images. Defaults to (480, 480).

        Returns:
            np.ndarray: The undistorted image.
        """
        try:
            calibration_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data/calibration/camera_intrinsics/{calibration_filename}')
            self.load_camera_parameters(calibration_path)

            for image_name, image in self.raw_images():
                
                h, w = image.shape[:2]
                    
                new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), balance)
                undistorted_image = cv2.undistort(image, self.K, self.D, None, new_K)
                    

                if show_image:
                    cv2.imshow("Undistorted Image", cv2.resize(undistorted_image, window_size))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                if save_image:
                    if output_path is None:
                        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data/undistorted_images/{image_name}')
                    cv2.imwrite(output_path, undistorted_image)
                    print(f"Undistorted image saved to {output_path}")
                
                return undistorted_image
            
        except Exception as e:
            print(f"Error undistorting image {image_name}: {e}")
            return None
        
    def export_camera_params_colmap(self, calibration_path: str = None) -> None:
        """
        Exports camera parameters in COLMAP format.
        """
        
        if calibration_path is None:
            calibration_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../../data/calibration/camera_intrinsics/pinhole_calibration.json')
        
        self.load_camera_parameters(calibration_path)
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        k1 = self.D[0][0]
        k2 = self.D[0][1]
        p1 = self.D[0][2]
        p2 = self.D[0][3]
        k3 = self.D[0][4]
        print('Camera parameters for COLMAP:')
        print(f"OPENCV: {fx}, {fy}, {cx}, {cy}, {k1}, {k2}, {p1}, {p2}")
    
            
        

if __name__ == '__main__':
    
    print('Running calibration script')
    
    ARUCO_DICT = cv2.aruco.DICT_5X5_100
    SQUARES_VERTICALLY = 12
    SQUARES_HORIZONTALLY = 9
    SQUARE_LENGTH = 0.06
    MARKER_LENGTH = 0.045
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

    instaCam = FisheyeCalibrator(
        ARUCO_DICT, SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, SQUARE_LENGTH, MARKER_LENGTH, 
        calibration_images_dir= os.path.join(CURRENT_PATH,'../.../data/calibration/images/'),
        raw_images_dir= os.path.join(CURRENT_PATH,'../../data/raw_images/descent_1')
        )
    
    instaCam.calibrate()
    for name, image in instaCam.raw_images():
        instaCam.undistort_image(image, image_name = name,show_image=False)

    
