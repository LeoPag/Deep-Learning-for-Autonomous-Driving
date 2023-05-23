import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import os
from load_data import load_data

data_path = os.path.join('data','data.p')
data = load_data(data_path)

vel = data["velodyne"]

#OBJECTS DO NOT FLY
#vel = vel[vel[:,2]<2]

x_y_int=np.delete(vel, 2, 1)

xcoord = x_y_int[:,0]
ycoord = x_y_int[:,1]

x_min = np.amin(xcoord)
y_min = np.amin(ycoord)
x_max = np.amax(xcoord)
y_max = np.amax(ycoord)


#Number of Intervals in each dimension
dim1 = int(np.ceil((x_max-x_min)/.2))
dim2 = int(np.ceil((y_max-y_min)/.2))

#Initializing output image
new_points = np.zeros((dim1,dim2))

#Filling new_points reflectance matrix
for point in x_y_int:
  x = point[0]-x_min
  y = point[1]-y_min
  ref = point[2]
  x_new = int(np.floor(x/.2))
  y_new = int(np.floor(y/.2))
  #Update matrix if new max reflectance is found
  if new_points[x_new,y_new]<ref:
    new_points[x_new,y_new] = ref

#Rotate image
new_points = ndimage.rotate(new_points, 90)

img=plt.imshow(new_points,cmap='gray')
plt.show()
