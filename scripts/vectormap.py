import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Camera parameters
K = np.array([[1332.91, 0, 1603.96], [0, 1350.38, 1610.60], [0, 0, 1]])  # Intrinsic matrix
D = np.array([-0.2231, 0.0339, 0.0020, -0.0001, -0.0019])  # Distortion coefficients for pinhole camera
K_fisheye = np.array([[993.52, 0, 1608.74], [0, 994.31, 1630.61], [0, 0, 1]])  # Intrinsic matrix for fisheye
D_fisheye = np.array([0.0324, -0.0187, 0.0011, -0.0002])  # Distortion coefficients for fisheye camera

# Image dimensions
width, height = 3264, 3264

# Generate grid of points
step = 100  # Adjust for resolution of the grid
x, y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
points = np.vstack([x.ravel(), y.ravel()]).T

def calculate_distortion(K, D, points, fisheye=False):
    points_cam = np.dot(np.linalg.inv(K), np.hstack([points, np.ones((points.shape[0], 1))]).T).T[:, :3]
    if fisheye:
        distorted_points = cv2.fisheye.projectPoints(points_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0]
    else:
        distorted_points = cv2.projectPoints(points_cam, np.zeros(3), np.zeros(3), K, D)[0]
    distorted_points = distorted_points.reshape(-1, 2)
    vectors = distorted_points - points
    return points, vectors

def plot_distortion(ax, points, vectors, title, norm, cmap):
    # Calculate vector magnitudes
    magnitudes = np.linalg.norm(vectors, axis=1)
    
    # Plot the quiver
    quiver = ax.quiver(points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1],
                       magnitudes, cmap=cmap, norm=norm,
                       angles='xy', scale_units='xy', scale=1)
    
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    
    return quiver

# Calculate distortion for both models
points_standard, vectors_standard = calculate_distortion(K, D, points)
points_fisheye, vectors_fisheye = calculate_distortion(K_fisheye, D_fisheye, points, fisheye=True)

# Calculate global min and max magnitudes
magnitudes_standard = np.linalg.norm(vectors_standard, axis=1)
magnitudes_fisheye = np.linalg.norm(vectors_fisheye, axis=1)
global_min = min(magnitudes_standard.min(), magnitudes_fisheye.min())
global_max = max(magnitudes_standard.max(), magnitudes_fisheye.max())

# Create a colormap and normalizer
cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=global_min, vmax=global_max)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

quiver1 = plot_distortion(ax1, points_standard, vectors_standard, "OpenCV Pinhole Distortion Vector Map", norm, cmap)
quiver2 = plot_distortion(ax2, points_fisheye, vectors_fisheye, "OpenCV Fisheye Distortion Vector Map", norm, cmap)

# Add a single colorbar for both plots
cbar = fig.colorbar(quiver1, ax=[ax1, ax2], label='Distortion Magnitude (pixels)')


plt.show()
plt.savefig('distortion_map.png')