import numpy as np
import copy
import random
import time
from utils.task1 import label2corners


def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie
       in each enlarged bounding box. Each ROI should contain exactly 512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (K,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''

    # a)

    def enlarge(pred, delta):

        enlarged_pred = copy.copy(pred)

        enlarged_pred[:, 3] += 2 * delta
        enlarged_pred[:, 4] += 2 * delta
        enlarged_pred[:, 5] += 2 * delta

        return enlarged_pred

    delta = config['delta']
    enlarged_pred = enlarge(pred, delta)

    # b)

    def get_axis_aligned(points, box_lab):
        '''
        Input
            points(N,3): 3D coordinates of N points
            box_lab(7,): bbox labels (x,y,z,h,w,l,ry)
        Output
            points_axis_al(N,3): 3D coordinates of N points in bbox axis-aligned reference frame
        '''

        box_cen = box_lab[:3]
        ry = box_lab[6]

        points_axis_al = points - box_cen

        rot = np.array([
            [np.cos(ry), 0, -np.sin(ry)],
            [0, 1, 0],
            [np.sin(ry), 0, np.cos(ry)]
        ])

        points_axis_al = np.transpose(rot @ np.transpose(points_axis_al))

        return points_axis_al

    def is_in_box(points_axis_al, box_lab):
        '''
        Input
            points_axis_al(N,3): 3D coordinates of N points in bbox axis-aligned reference frame
            box_lab(7,): bbox labels (x,y,z,h,w,l,ry)
        Output
            flag(N,): True for points inside the bbox
        '''

        h = box_lab[3]
        w = box_lab[4]
        l = box_lab[5]

        flag = (points_axis_al[:, 0] >= -l / 2) & (points_axis_al[:, 0] <= l / 2) \
               & (points_axis_al[:, 1] >= -h) & (points_axis_al[:, 1] <= 0) \
               & (points_axis_al[:, 2] >= -w / 2) & (points_axis_al[:, 2] <= w / 2)

        return flag

    pool_indexes = []
    valid_indexes = []

    rng = np.random.default_rng()

    for box_idx in range(enlarged_pred.shape[0]):

        box_lab = enlarged_pred[box_idx, :]

        xyz_axis_al = get_axis_aligned(xyz, box_lab)
        mask = is_in_box(xyz_axis_al, box_lab)
        inliers_idx = np.flatnonzero(mask)
        inliers_xyz = xyz[inliers_idx, :]
        num_inliers = inliers_xyz.shape[0]

        if num_inliers == 0:
            continue

        elif num_inliers < config['max_points']:

            additional_indexes = rng.choice(inliers_idx, config['max_points'] - num_inliers, replace=True)
            inliers_idx = np.concatenate((inliers_idx, additional_indexes))
            pool_indexes.append(inliers_idx)

        elif num_inliers > config['max_points']:

            inliers_idx = rng.choice(inliers_idx, config['max_points'], replace=False)
            pool_indexes.append(inliers_idx)

        elif num_inliers == config['max_points']:
            pool_indexes.append(inliers_idx)

        valid_indexes.append(box_idx)

    valid_indexes = np.array(valid_indexes)  # [K']
    pool_indexes = np.array(pool_indexes)  # [K'xM]

    pooled_xyz = np.zeros((valid_indexes.shape[0], config['max_points'], 3))
    pooled_feat = np.zeros((valid_indexes.shape[0], config['max_points'], feat.shape[1]))

    pooled_xyz = xyz[pool_indexes, :]
    pooled_feat = feat[pool_indexes, :]
    valid_pred = pred[valid_indexes, :]

    return valid_pred, pooled_xyz, pooled_feat
