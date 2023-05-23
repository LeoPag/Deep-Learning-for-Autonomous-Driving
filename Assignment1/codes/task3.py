import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import os
import colorsys
import cv2 as cv
from load_data import load_data
from data_utils import line_color

data_path = "data/data.p"

data = load_data(data_path)
points = data["velodyne"]

image = data["image_2"]

intrinsic = data["K_cam2"]
extrinsic = data["T_cam2_velo"]
sem_label = data["sem_label"]
color_map = data["color_map"]
labels = data["labels"]


#Intensity becomes fourth homeogeneous coordinate
points[:,3] = 1

#Recovering x,y,z velodyne coordinates
x_coord = points[:,0]
y_coord = points[:,1]
z_coord = points[:,2]

#Computing points' distances from velodyne
distance = np.sqrt(x_coord**2 + y_coord**2)

#Computing elevation angle
angles = np.arctan2(z_coord, distance)

#Computing max_angle and min_angle to perform uniform discretization
max_angle = np.amax(angles)
min_angle = np.amin(angles)

#Defining 3x4 matrix to perform correct projection
mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

#Switching to CAM2 reference frame
cam2_points = np.transpose(extrinsic @ np.transpose(points))

#Filtering points behind camera 2
angles = angles[cam2_points[:,2] >= 0]
points = points[cam2_points[:,2] >= 0]
sem_label = sem_label[cam2_points[:,2] >= 0]
cam2_points = cam2_points[cam2_points[:,2] >= 0]

#Switching to pixel coordinates (CAM2) by projecting points
proj_points = np.transpose(intrinsic @ mat @ np.transpose(cam2_points))

#Normalizing each point to unary third coordinate
div=proj_points[:,2]
div=np.reshape(div,(proj_points.shape[0],1))
normalized = np.divide(proj_points,div)
normalized = normalized[:,:2]

#Computing angular resolution provided by the lasers
number_laser = 64
res = (max_angle - min_angle)/ number_laser


#Computing laser ID for each point
laser_ids = np.ceil((angles - min_angle) / res)


h_values = line_color(laser_ids)
h_values = np.array(h_values)
h_values = h_values*0.75/np.amax(h_values) #so that the h values are roughly 0, 0.25, 0.5, 0.75 (0°, 90°, 180°, 270°)
h_values = np.reshape(h_values,(h_values.shape[0],1))
hsv_values = np.hstack((h_values, np.ones((h_values.shape[0],2))))
rgb_values = cl.hsv_to_rgb(hsv_values)

plt.scatter(normalized[:,0],normalized[:,1], c = rgb_values, s = 1)
plt.imshow(image)

plt.show()
