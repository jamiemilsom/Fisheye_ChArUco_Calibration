import numpy as np
import os
import scipy.io

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

image_points_path = '../data/calibration/image_points.npy'
object_points_path = '../data/calibration/object_points.npy'

image_points = np.load(os.path.join(CURRENT_PATH,image_points_path)).reshape(-1,88,2)
object_points = np.load(os.path.join(CURRENT_PATH,object_points_path)).reshape(-1,88,3)

print('OpenCV FORMAT')
print('Image points shape:', image_points.shape)
print('Object points shape:', object_points.shape)

# MATLAB FORMAT
    # imagePoints = num_corners x 2 x num_images
    # worldPoints = num_corners x 2

object_points = object_points[0,:,0:2]

image_points = np.transpose(image_points, (1, 2, 0))


object_points = object_points.astype(np.double)
image_points = image_points.astype(np.double)
print('MATLAB FORMAT')
print('Image points shape:', image_points.shape)
print('Object points shape:', object_points.shape)


scipy.io.savemat(os.path.join(CURRENT_PATH,'../data/calibration/calibration_points.mat'), {'imagePoints': image_points, 'worldPoints': object_points})



