import numpy as np
import matplotlib.pyplot as plt
import os
from load_data import load_data
import matplotlib.patches as patches

data_path = "data/data.p"

data = load_data(data_path)

points = data["velodyne"]

image = data["image_2"]

intrinsic = data["K_cam2"]
extrinsic = data["T_cam2_velo"] #From Velo frame to CAM2 frame
sem_label = data["sem_label"]
color_map = data["color_map"]
labels = data["labels"]


"""
2.1
"""
#points = velodyne[velodyne[:,0] >= 0]
#sem_label = sem_label[velodyne[:,0] >= 0]

points[:,3] = 1 #Intensity becomes Homeogeneous coordinates

mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

cam2_points = np.transpose(extrinsic @ np.transpose(points))


#Filtering points behind camera 2

sem_label = sem_label[cam2_points[:,2] >= 0]
cam2_points = cam2_points[cam2_points[:,2] >= 0]

#Pixel coordinate (CAM2)
proj_points = np.transpose(intrinsic @ mat @ np.transpose(cam2_points))


#Normalizing each point to unary third coordinate
div=proj_points[:,2]
div=np.reshape(div,(proj_points.shape[0],1))
normalized = np.divide(proj_points,div)
normalized = normalized[:,:2]

#Creating colors array
colors = []

for elem in sem_label:
    number = elem[0]
    colors.append(color_map[number])
colors = np.array(colors)

#From BGR to RGB
colors[:,[0, 2]] = colors[:,[2, 0]]






"""
2.2
"""
cars = data["objects"]
T_0_velo = data["T_cam0_velo"]
points_bb = []

fig, ax = plt.subplots()

for car in cars:
    h = car[8]  #HEIGHT
    w = car[9]  #WIDTH
    l = car[10] #LENGTH
    x_cen = car[11]
    y_cen = car[12]
    z_cen = car[13]

    ry = car[14]

    blb = np.array([-l/2, 0, w/2, 1])
    brb = np.array([-l/2, 0, -w/2, 1])
    brf = np.array([l/2, 0, -w/2, 1])
    blf = np.array([l/2, 0, w/2, 1])
    tlb = np.array([-l/2, -h, w/2, 1])
    trb = np.array([-l/2, -h, -w/2, 1])
    trf = np.array([l/2, -h, -w/2, 1])
    tlf = np.array([l/2, -h, w/2, 1])
   
    points_car = np.vstack((blb,brb,brf,blf,tlb,trb,trf,tlf))
    x = np.reshape(points_car[:,0],(8,1))
    y = np.reshape(points_car[:,1],(8,1))
    z = np.reshape(points_car[:,2],(8,1))
    
    x_tr = x*np.cos(ry) + z*np.sin(ry)
    z_tr = z*np.cos(ry) - x*np.sin(ry)

    points_car_tr = np.hstack((x_tr,y,z_tr,np.ones((points_car.shape[0],1))))
    points_car_tr += np.array([x_cen, y_cen, z_cen, 1])
    
    points_img_2 = np.transpose(intrinsic @ mat @ extrinsic @ np.linalg.inv(T_0_velo) @ np.transpose(points_car_tr))
    points_bb.append(points_img_2)

    div=points_img_2[:,2]
    div=np.reshape(div,(points_img_2.shape[0],1))
    points_img_2 = np.divide(points_img_2,div)
    points_rect1 = points_img_2[:4,:2]
    rect1 = patches.Polygon(points_rect1,fill=False,color='#03fc2c',linewidth=2)
    points_rect2 = points_img_2[4:,:2]
    rect2 = patches.Polygon(points_rect2,fill=False,color='#03fc2c',linewidth=2)
    points_rect3 = points_img_2[[0,1,5,4],:2]
    rect3 = patches.Polygon(points_rect3,fill=False,color='#03fc2c',linewidth=2)
    points_rect4 = points_img_2[[2,3,-1,-2],:2]
    rect4 = patches.Polygon(points_rect4,fill=False,color='#03fc2c',linewidth=2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)

plt.scatter(normalized[:,0],normalized[:,1], c = colors/255.0, s=0.5)
ax.imshow(image)
plt.show()
