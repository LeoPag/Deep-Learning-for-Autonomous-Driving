# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)

        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])

    def update(self, points):
        '''
        :param points: point cloud data
                        shape (N, 3)
        Task 2: Change this function such that each point
        is colored depending on its semantic label

        '''

        #Retrieving data

        color_map = data['color_map']
        sem_label = data['sem_label']
        points = data["velodyne"]
        image = data["image_2"]
        intrinsic = data["K_cam2"]
        extrinsic = data["T_cam2_velo"] #From Velo frame to CAM2 frame
        sem_label = data["sem_label"]
        color_map = data["color_map"]
        labels = data["labels"]

        #Computing image dimensions
        dim_x = image.shape[1]
        dim_y = image.shape[0]

        """COMMENT THIS ENTIRE BLOCK FOR FULL VISUALIZATION. BY UNCOMMENTING, ONLY CAMERA FOV IS VISUALIZED
        #Intensity becomes fourth homeogeneous coordinate
        points[:,3] = 1

        #Defining 3x4 matrix to perform correct projection
        mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

        #Switching to CAM2 reference frame
        cam2_points = np.transpose(extrinsic @ np.transpose(points))

        #Filtering points behind camera 2
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

        #Filtering points outside CAM2 FOV - COMMENT THESE 2 LINES FOR FULL VISUALIZATION
        sem_label = sem_label[(0 <= normalized[:,0]) & (normalized[:,0] <= dim_x) & (0 <= normalized[:,1]) & (normalized[:,1] <= dim_y)]
        points = points[(0 <= normalized[:,0]) & (normalized[:,0] <= dim_x) & (0 <= normalized[:,1]) & (normalized[:,1] <= dim_y)]
        """
        #Creating colors array
        colors = []

        for elem in sem_label:
            number = elem[0]
            colors.append(color_map[number])

        colors = np.array(colors)

        #From BGR to RGB
        colors[:,[0, 2]] = colors[:,[2, 0]]

        #Plotting the coloured point cloud restricted to FOV
        self.sem_vis.set_data(points[:,:3], size=3, edge_color = colors/255.0)

    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

if __name__ == '__main__':
    data = load_data('data/data.p') # Change to data.p for your final submission
    visualizer = Visualizer()

    visualizer.update(data["velodyne"][:,:3])
    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''

    cars = data["objects"]
    T_0_velo = data["T_cam0_velo"]

    #Array containing coordinates of all cars' bb-corners
    all_corners = []

    for car in cars:
        #Dimensions and center coordinates of bounded boxes in CAM0 frame
        h = car[8]  #HEIGHT
        w = car[9]  #WIDTH
        l = car[10] #LENGTH
        x_cen = car[11]
        y_cen = car[12]
        z_cen = car[13]

        #Rotation angle of bounded box around y-CAM0
        ry = car[14]

        #Coordinates of bounded box corners in C-frame(see Report)
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

        corners = np.transpose(np.linalg.inv(T_0_velo) @ np.transpose(points_car_tr))

        #Normalizing each point to unary fourth coordinate
        div=corners[:,3]
        div=np.reshape(div,(corners.shape[0],1))
        corners = np.divide(corners,div)

        #Discarding fourth homogeneous coordinate
        corners = corners[:,:3]
        all_corners.append(corners)

    all_corners = np.array(all_corners)

    #Calling update boxes
    visualizer.update_boxes(all_corners)
    vispy.app.run()
