# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 14:41
# @Author  : WeiHuang


import torch
import torch.nn as nn
import utils.array_tool as at
import numpy as np

class RPN_Label(object):

    def __init__(self, n_sample=256, pos_ratio=0.5, pos_thresh=0.7, neg_thresh=0.3):
        super(RPN_Label, self).__init__()
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh


    def __call__(self, anchor, gt_box, img_size):
        h, w = img_size
        anchor = at.toNumpy(anchor)
        gt_box = at.toNumpy(gt_box)

        iou = at.twobox_iou(anchor, gt_box)
        valid = np.where((anchor[:, 0]>=0) & (anchor[:, 1]>=0) & (anchor[:, 2]<=h) & anchor[:, 3]<=w)[0]
        label = np.empty(len(valid), dtype=np.int32)
        label.fill(-1)
        valid_iou = iou[valid]
        max_iou = valid_iou.max(axis=1) # N
        gt_max_iou = valid_iou.max(axis=0)
        gt_max_iou_index = np.where(valid_iou==gt)

        pos_index = np.where(max_iou>=self.pos_thresh)[0]
        neg_index = np.where(max_iou<self.neg_thresh)[0]
        n_pos = int(self.n_sample * self.pos_ratio)
        n_pos = min(n_pos, len(pos_index))
        n_neg = self.n_sample - n_pos
        n_neg = min(n_neg, len(neg_index))
        #
        pos_index = np.random.choice(pos_index, n_pos, replace=False)
        neg_index = np.random.choice(neg_index, n_neg, replace=False)
        torch.o





class Head_Label(nn.Module):

    def __init__(self):
        super(Head_Label, self).__init__()

    def forward(self):
        pass
