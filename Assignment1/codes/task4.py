import numpy as np
import matplotlib.pyplot as plt
import os
from load_data import load_data
from data_utils import compute_timestamps, load_oxts_velocity, load_oxts_angular_rate, load_from_bin, calib_velo2cam, calib_cam2cam, depth_color
import matplotlib.colors as cl
from scipy.spatial.transform import Rotation


#####**CHOOSE FRAME**#####

frame = '0000000037'

#Retrieving data
image = plt.imread('data/problem_4/image_02/data/' + frame + '.png')
points = load_from_bin('data/problem_4/velodyne_points/data/' + frame + '.bin')
velocities = load_oxts_velocity('data/problem_4/oxts/data/' + frame + '.txt')
ang_vel = load_oxts_angular_rate('data/problem_4/oxts/data/' + frame + '.txt')
start_time = compute_timestamps('data/problem_4/velodyne_points/timestamps_start.txt',frame)
end_time = compute_timestamps('data/problem_4/velodyne_points/timestamps_end.txt',frame)
front_time = compute_timestamps('data/problem_4/velodyne_points/timestamps.txt',frame)
gps_time  = compute_timestamps('data/problem_4/oxts/timestamps.txt',frame)
image_time = compute_timestamps('data/problem_4/image_02/timestamps.txt',frame)
R,t = calib_velo2cam('data/problem_4/calib_velo_to_cam.txt')
cam_to_cam = calib_cam2cam('data/problem_4/calib_cam_to_cam.txt','02')
R_imu_to_velo, t_imu_to_velo = calib_velo2cam('data/problem_4/calib_imu_to_velo.txt')

T_velo_to_cam = np.hstack((R,t))
T_velo_to_cam = np.vstack((T_velo_to_cam,np.array([0,0,0,1])))
T_imu_to_velo = np.hstack((R_imu_to_velo,-R_imu_to_velo@t_imu_to_velo))
T_imu_to_velo = np.vstack((T_imu_to_velo,np.array([0,0,0,1])))
T_velo_to_imu = np.linalg.inv(T_imu_to_velo)


#Defining LIDAR angular velocity
omega = 10 * (2*np.pi)

#Recovering x,y,z velodyne coordinates
x_coord = points[:,0]
y_coord = points[:,1]
z_coord = points[:,2]

#Defining 3D-point in non-homogeneous and homogeneous coordinates. Points coordinates are affected by distortion
nonhom_3d = np.transpose(np.vstack((x_coord,y_coord,z_coord)))
hom_3d = np.hstack((nonhom_3d,np.ones((nonhom_3d.shape[0],1))))


#####**COLORS**#####

dist = np.sqrt((x_coord**2 + y_coord**2 + z_coord**2))
h_values = depth_color(dist)
h_values = np.array(h_values)
h_blue = 240
h_values = (h_values - np.amin(h_values))*(h_blue/360)/np.amax(h_values) #from red to blue
h_values = np.reshape(h_values,(h_values.shape[0],1))
hsv_values = np.hstack((h_values, np.ones((h_values.shape[0],2))))
rgb_values = cl.hsv_to_rgb(hsv_values)
#rgb_values[:,[0, 2]] = rgb_values[:,[2, 0]]



#####**TRANSLATION CORRECTION**#####

# Retrieving forward and leftward velocity
vel_x = velocities[0]
vel_y = velocities[1]

# Transferring points to IMU/GPS frame
hom_3d_imu = np.transpose(T_velo_to_imu @ np.transpose(hom_3d))

# Correcting the translation distortion---SEE REPORT TO UNDERSTAND THE REASONING
x_coord_imu = hom_3d_imu[:,0]
y_coord_imu = hom_3d_imu[:,1]
z_coord_imu = hom_3d_imu[:,2]

x_undistorted_imu = x_coord_imu + vel_x*(np.arctan2(-y_coord_imu,x_coord_imu))/omega
y_undistorted_imu = y_coord_imu + vel_y*(np.arctan2(-y_coord_imu,x_coord_imu))/omega
z_undistorted_imu = z_coord_imu


#3D-point in non-homogeneous coordinates. New coordinates take into account distortion caused by translation
nonhom_3d_imu = np.transpose(np.vstack((x_undistorted_imu,y_undistorted_imu,z_undistorted_imu)))


#####**ROTATION CORRECTION**#####

# Retrieving angular rate around upward axis
omega_car = ang_vel[2]

# fi0 is a scalar defining the angle between the front of the car and the direction of the Lidar at the start of the scan
fi0 = omega*(front_time-start_time)

# fi_det is a vector containing the angles between the points and the front of the car when the points are scanned by the LIDAR
fi_det = np.arctan2(y_coord,x_coord)

# delta_t is a vector containing the time interval between the start of the LIDAR scan and the detection  of each point.
delta_t = (fi0 - fi_det)/omega

# fi_car is a vector containing the angular displacement of the car after delta_t
fi_car = omega_car * delta_t


# Defining rotation matrices performing rotation of (-fi_car) around z-axis ---SEE REPORT TO UNDERSTAND THE REASONING
r_car = Rotation.from_euler('z', -fi_car, degrees=False)
rot_car = r_car.as_matrix()

# Angle_foto is a scalar defining the angular displacement between the starting time of the scan and the photo time
angle_foto = (front_time-start_time)*omega_car

# Defining a rotation matrix performing rotation of (angle_foto) around z-axis ---SEE REPORT TO UNDERSTAND THE REASONING
r_foto = Rotation.from_euler('z', angle_foto, degrees = False)
rot_foto = r_foto.as_matrix()

#3D-point in non-homogeneous coordinates. New coordinates take into account distortion caused by rotation
nonhom_3d_imu = np.reshape(nonhom_3d_imu,((nonhom_3d_imu.shape[0]),1,3))
nonhom_3d_imu = np.squeeze(nonhom_3d_imu@rot_car)
nonhom_3d_imu = np.squeeze(nonhom_3d_imu@rot_foto)



#####**PROJECTION**#####

# Transferring back to velodyne frame
hom_3d_imu = np.hstack((nonhom_3d_imu,np.ones((nonhom_3d_imu.shape[0],1))))
hom_3d = np.transpose(T_imu_to_velo @ np.transpose(hom_3d_imu))

rgb_values = rgb_values[hom_3d[:,0] > 0]
hom_3d = np.transpose(hom_3d[hom_3d[:,0] > 0])

image_points = np.transpose(cam_to_cam @ T_velo_to_cam @ hom_3d)

#Normalizing each point to unary third coordinate
div=image_points[:,2]
div=np.reshape(div,(image_points.shape[0],1))
normalized = np.divide(image_points,div)
normalized = normalized[:,:2]



#####**PRINTOUT**#####

plt.scatter(normalized[:,0],normalized[:,1], c = rgb_values, s = 1)
plt.imshow(image)
plt.show()
