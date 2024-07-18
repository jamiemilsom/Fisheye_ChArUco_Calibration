import os
import numpy as np
import cv2


# ENTER YOUR REQUIREMENTS HERE:
ARUCO_DICT = cv2.aruco.DICT_5X5_250
SQUARES_VERTICALLY = 12
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.01
MARKER_LENGTH = 0.0075
# ...
PATH_TO_YOUR_IMAGES = '/home/jamie/Documents/reconstruction/data/calibration/jpg'
OUTPUT_PATH = '/home/jamie/Documents/reconstruction/data/calibration/undistorted'

# ------------------------------


import os
import numpy as np
import cv2


# ENTER YOUR REQUIREMENTS HERE:
ARUCO_DICT = cv2.aruco.DICT_5X5_250
SQUARES_VERTICALLY = 12
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.01
MARKER_LENGTH = 0.0075
# ...
PATH_TO_YOUR_IMAGES = '/home/jamie/Documents/reconstruction/data/calibration/jpg'
# ------------------------------


def visualize_aruco_markers(image_files, resize_factor=0.5):
  """
  This function iterates through image files, detects ArUco markers,
  and displays the image with markers highlighted at a reduced size.

  Args:
      image_files: A list of paths to image files.
      resize_factor (optional): A float between 0 and 1 to scale the image size.
  """
  
  # Define the Aruco dictionary
  aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
  params = cv2.aruco.DetectorParameters()

  for image_file in image_files:

    image = cv2.imread(image_file)

    # Convert to grayscale (optional, but might improve marker detection)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=params)

    # If markers are detected, draw them on the image    
    if marker_ids is not None and len(marker_ids) > 0:
      cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
      print(f"Found {len(marker_ids)} markers in {image_file}")

    # Resize the image before displaying
    (h, w) = image.shape[:2]
    new_width = int(w * resize_factor)
    new_height = int(h * resize_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Display the resized image with markers (or the original image if no markers found)
    cv2.imshow('Detected ArUco Markers', resized_image)
    cv2.waitKey(0)


def calibrate_and_save_parameters():
    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".jpg")]
    image_files.sort()

    all_charuco_corners = []
    all_charuco_ids = []

    for image_file in image_files:
        image = cv2.imread(image_file)
        image_copy = image.copy()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
        
        # If at least one marker is detected
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                
    print('Starting calibration...')
    # Calibrate camera
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image.shape[:2], None, None)
    print('Saving calibration data...')
    # Save calibration data
    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)

    print('Making output')
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create directory if it doesn't exist
    print('Saving new images')
    # Iterate through undistorting and saving images
    for image_file in image_files:
        image = cv2.imread(image_file)
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

        # Extract filename without extension
        filename, _ = os.path.splitext(os.path.basename(image_file))
        output_filename = os.path.join(OUTPUT_PATH, f"{filename}_undistorted.jpg")

        # Save the undistorted image
        cv2.imwrite(output_filename, undistorted_image)

    cv2.destroyAllWindows()

calibrate_and_save_parameters()

# visualize_aruco_markers(image_files,0.2)