import numpy as np
from shapely.geometry import Polygon


def label2corners(label):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    '''

    if len(label.shape) == 1:
        label = np.expand_dims(label, 0)

    x_cen = label[:, 0]
    y_cen = label[:, 1]
    z_cen = label[:, 2]
    h = label[:, 3]
    w = label[:, 4]
    l = label[:, 5]
    ry = label[:, 6]

    N = label.shape[0]
    zero = np.zeros((N,))

    blb = np.transpose(np.vstack((-l/2, zero, w/2)))
    brb = np.transpose(np.vstack((-l/2, zero, -w/2)))
    brf = np.transpose(np.vstack((l/2, zero, -w/2)))
    blf = np.transpose(np.vstack((l/2, zero, w/2)))
    tlb = np.transpose(np.vstack((-l/2, -h, w/2)))
    trb = np.transpose(np.vstack((-l/2, -h, -w/2)))
    trf = np.transpose(np.vstack((l/2, -h, -w/2)))
    tlf = np.transpose(np.vstack((l/2, -h, w/2)))

    corners = np.stack((blb, brb, brf, blf, tlb, trb, trf, tlf), axis=1)  # corners expressed in the bb-centered rf [Nx8x3]

    x_corn = corners[:, :, 0]  # x coordinate of corners of all points [Nx8]
    y_corn = corners[:, :, 1]
    z_corn = corners[:, :, 2]

    cosine = np.diag(np.cos(ry))  # [NxN]
    sine = np.diag(np.sin(ry))

    # perform rotation
    x_tr = np.transpose(np.transpose(x_corn) @ cosine) + np.transpose(np.transpose(z_corn) @ sine)  # [Nx8]
    z_tr = np.transpose(np.transpose(z_corn) @ cosine) - np.transpose(np.transpose(x_corn) @ sine)

    # perform translation
    x_tr += np.transpose(np.tile(x_cen, (8, 1)))  # [Nx8]
    y_tr = y_corn + np.transpose(np.tile(y_cen, (8, 1)))
    z_tr += np.transpose(np.tile(z_cen, (8, 1)))

    corners = np.dstack((x_tr, y_tr, z_tr))  # [Nx8x3] in CAM0 frame

    return corners


def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners (x,y,z,h,w,l,ry)
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''

    N = pred.shape[0]
    M = target.shape[0]

    pred_corn = label2corners(pred)
    targ_corn = label2corners(target)

    iou = np.zeros((N, M))

    for i in range(N):
        prediction = pred_corn[i, :, :]
        pred_base = Polygon(prediction[:4, [0, 2]])
        pred_y_min = np.amin(prediction[:, 1])
        pred_h = pred[i, 3]
        pred_vol = pred_base.area * pred_h
        for j in range(M):
            targ = targ_corn[j, :, :]
            targ_base = Polygon(targ[:4, [0, 2]])
            inter_area = pred_base.intersection(targ_base).area
            if inter_area != 0:
                targ_y_min = np.amin(targ[:, 1])
                targ_h = target[j, 3]
                targ_vol = targ_base.area * targ_h
                h_lo = np.maximum(pred_y_min, targ_y_min)
                h_hi = np.minimum(pred_y_min+pred_h, targ_y_min+targ_h)
                inter_vol = inter_area * np.maximum(0, h_hi-h_lo)
                union_vol = pred_vol + targ_vol - inter_vol
                iou[i, j] = inter_vol / union_vol

    return iou


def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''

    iou = get_iou(pred, target)
    iou = np.amax(iou, axis=0)  # so that at most 1 pred is associated to each target

    iou = iou[iou >= threshold]
    recall = iou.shape[0]/target.shape[0]

    return recall
