import numpy as np
import copy
from shapely.geometry import Polygon

from utils.task1 import get_iou
from utils.task1 import label2corners


def get_iou_2d(pred, target):
    '''
    input
        pred (N,7) 3D bounding box corners (x,y,z,h,w,l,ry)
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 2D intersection-over-union
    '''

    N = pred.shape[0]
    M = target.shape[0]

    pred_corn = label2corners(pred)
    targ_corn = label2corners(target)

    iou = np.zeros((N, M))

    for i in range(N):
        prediction = pred_corn[i, :, :]
        pred_base = Polygon(prediction[:4, [0, 2]])
        pred_area = pred_base.area
        for j in range(M):
            targ = targ_corn[j, :, :]
            targ_base = Polygon(targ[:4, [0, 2]])
            inter_area = pred_base.intersection(targ_base).area
            if inter_area != 0:
                targ_area = targ_base.area
                union_area = pred_area + targ_area - inter_area
                iou[i, j] = inter_area / union_area
    return iou


def nms(pred, score, threshold):
    '''
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    '''

    s_f = []
    c_f = []
    
    while pred.shape[0] != 0:

        i = np.argmax(score)
        d_i = pred[i, :].astype('float64')
        score_i = score[i].astype('float64')
        pred = np.delete(pred, i, axis=0)
        score = np.delete(score, i)

        s_f.append(d_i)
        c_f.append(score_i)

        d_i = np.expand_dims(d_i, 0)

        iou = get_iou_2d(pred, d_i)
        iou = np.squeeze(iou, axis=1)

        mask_to_keep = iou < threshold
        pred = pred[mask_to_keep, :]
        score = score[mask_to_keep]

    s_f = np.array(s_f)
    c_f = np.array(c_f)
    if len(c_f.shape) < 2:
        c_f = np.expand_dims(c_f, 1)

    return s_f, c_f
