import os
import numpy as np
import cv2

PATH_TO_YOUR_IMAGES = '/home/jamie/Documents/reconstruction/data/descent_jpgs_2024-01-10/descent_jpgs/descent_1'
OUTPUT_PATH = '/home/jamie/Documents/reconstruction/data/descent_jpgs_2024-01-10/descent_jpgs/descent_1_undistorted'

# ------------------------------

def load_camera_calibration():
  """
  Loads camera matrix and distortion coefficients from saved NPY files.

  Returns:
      A tuple containing the loaded camera matrix and distortion coefficients.
  """
  # Load camera matrix and distortion coefficients
  camera_matrix = np.load('camera_matrix.npy')
  dist_coeffs = np.load('dist_coeffs.npy')
  return camera_matrix, dist_coeffs


def undistort_images():
  """
  Undistorts images from the specified path and saves them to the output path.

  Uses the camera calibration data (camera_matrix and distortion coefficients) 
  loaded from .npy files.
  """
  # Load camera calibration
  camera_matrix, dist_coeffs = load_camera_calibration()

  # Load image file paths
  image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".jpg")]
  image_files.sort()

  # Create the output directory if it doesn't exist
  os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create directory if it doesn't exist

  for image_file in image_files:
    image = cv2.imread(image_file)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    # Extract filename without extension
    filename, _ = os.path.splitext(os.path.basename(image_file))
    output_filename = os.path.join(OUTPUT_PATH, f"{filename}_undistorted.jpg")

    # Save the undistorted image
    cv2.imwrite(output_filename, undistorted_image)

  cv2.destroyAllWindows()


undistort_images()
