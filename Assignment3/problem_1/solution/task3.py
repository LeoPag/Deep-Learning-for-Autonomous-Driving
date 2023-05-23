import copy

import numpy as np
import math

from .task1 import get_iou

def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (string) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples (32)
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples) (0.5)
    '''

    # a)

    pairwise_iou = get_iou(pred, target)
    iou = np.amax(pairwise_iou, axis=1)
    max_idx = np.argmax(pairwise_iou, axis=1)
    assigned_targets = target[max_idx, :]

    # b)

    def only_foreground(iou, config):
        """
        input
            iou (N+M,1): maximum IoU per proposal
            config: for IoU thresholds
        output
            flag (bool): True if the scene contains only foreground proposals
        """

        return not np.any(iou[:N] < config['t_fg_lb'])  # more efficient than np.all (as soon as it finds a non-foreground proposal it returns False)

    def only_background(iou, config):
        """
        input
            iou (N+M,1): maximum IoU per proposal
            config: for IoU thresholds
        output
            flag (bool): True if the scene contains only background proposals (excluding the additional foreground preds assigned through the "max per column" crtierion)
        """

        return not np.any(iou[:N] >= config['t_bg_up'])

    def only_easy_background(iou, config):
        """
        input
            iou (N+M,1): maximum IoU per proposal
            config: for IoU thresholds
        output
            flag (bool): True if the scene contains only easy background proposals (excluding the additional foreground preds assigned through the "max per column" crtierion)
        """

        return (only_background(iou, config)) & (not np.any(iou[:N] >= config['t_bg_hard_lb']))

    def only_hard_background(iou, config):
        """
        input
            iou (N+M,1): maximum IoU per proposal
            config: for IoU thresholds
        output
            flag (bool): True if the scene contains only hard background proposals (excluding the additional foreground preds assigned through the "max per column" crtierion)
        """

        return (only_background(iou, config)) & (not np.any(iou[:N] < config['t_bg_hard_lb'])) & (not np.any(iou[:N] >= config['t_bg_up']))

    def sample_indexes(arr, num):
        """
        input
            arr (list or ndarray): array of indexes (int)
            num (int): desired number of samples
        output
            sampled_idx (ndarray): sampled indexes
        """
        num = int(num)
        if len(arr) > num:
            sampled_idx = np.random.choice(arr, num, replace=False)
        elif len(arr) < num:
            additional_indexes = np.random.choice(arr, num-len(arr), replace=True)
            sampled_idx = np.concatenate((arr, additional_indexes))
        else:
            sampled_idx = np.array(arr)

        return sampled_idx

    def sample_background_indexes(num, iou, config):
        """
        input
            num (int): number of desired background samples
            iou (N+M,1): maximum IoU per proposal
            config: for IoU thresholds
        output
            sampled_idx (ndarray): sampled indexes
        """

        num_hard = math.floor(config['bg_hard_ratio'] * num)
        num_easy = num - num_hard
        hard_bg_idx = np.where((iou[:N] >= config['t_bg_hard_lb']) & (iou[:N] < config['t_fg_lb']))[0]
        if len(hard_bg_idx) != 0:
            sampled_hard_idx = sample_indexes(hard_bg_idx, num_hard)
        else:
            sampled_hard_idx = []
            num_easy = num
        easy_bg_idx = np.where(iou[:N] < config['t_bg_hard_lb'])[0]
        if len(easy_bg_idx) != 0:
            sampled_easy_idx = sample_indexes(easy_bg_idx, num_easy)
        else:
            sampled_easy_idx = []
            sampled_hard_idx = sample_indexes(hard_bg_idx, num)
        sampled_idx = np.concatenate((sampled_hard_idx, sampled_easy_idx))

        return sampled_idx

    num_samples = 64
    N = iou.shape[0]  # number of preds before "highest IoU for a ground truth" criterion (useful because after N there will surely be no background preds)

    # Add foreground preds according to the "highest IoU for a ground truth" criterion

    max_pred_idx = np.argmax(pairwise_iou, axis=0)
    for i in range(len(max_pred_idx)):
        new_assigned_target = target[i, :]
        assigned_targets = np.vstack((assigned_targets, new_assigned_target))
        idx = max_pred_idx[i]
        new_xyz = np.expand_dims(xyz[idx, :, :], 0)
        xyz = np.vstack((xyz, new_xyz))
        new_feat = np.expand_dims(feat[idx, :, :], 0)
        feat = np.vstack((feat, new_feat))
        new_iou = pairwise_iou[idx, i]
        iou = np.append(iou, new_iou)

    # Sample indexes according to rules

    if only_foreground(iou, config):
        indexes = sample_indexes(range(assigned_targets.shape[0]), num_samples)
    elif only_background(iou, config):
        if only_easy_background(iou, config):
            indexes = sample_indexes(range(N), num_samples)
        elif only_hard_background(iou, config):
            indexes = sample_indexes(range(N), num_samples)
        else:
            indexes = sample_background_indexes(num_samples, iou, config)
    else:
        fg_idx = np.concatenate(
            (np.where(iou[:N] >= config['t_fg_lb'])[0], np.arange(N, assigned_targets.shape[0])))
        num_fg = len(fg_idx)
        if num_fg > config['num_fg_sample']:
            sampled_fg_idx = sample_indexes(fg_idx, config['num_fg_sample'])
        else:
            sampled_fg_idx = fg_idx
        num_bg = np.maximum(num_samples-num_fg, config['num_fg_sample'])
        sampled_bg_idx = sample_background_indexes(num_bg, iou, config)
        indexes = np.concatenate((sampled_fg_idx, sampled_bg_idx))

    # Use indexes to get the desired outputs

    indexes = indexes.astype(int)
    assigned_targets = assigned_targets[indexes, :]
    xyz = xyz[indexes, :, :]
    feat = feat[indexes, :, :]
    iou = iou[indexes]

    return assigned_targets, xyz, feat, iou
