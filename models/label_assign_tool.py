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
        n_anchor = len(anchor)

        valid = np.where((anchor[:, 0]>=0) & (anchor[:, 1]>=0) & (anchor[:, 2]<=h) & (anchor[:, 3]<=w))[0]
        label = np.empty(len(valid), dtype=np.int32)
        label.fill(-1)
        anchor = anchor[valid]

        iou = at.twobox_iou(anchor, gt_box)
        max_iou = iou.max(axis=1) # N
        argmax_iou = iou.argmax(axis=1)
        gt_max_iou = iou.max(axis=0)
        gt_max_iou_index = np.where(iou==gt_max_iou)[0]

        # set label
        # first set neg label
        label[max_iou<self.neg_thresh] = 0
        # second set pos label because some iou is lower than neg_thresh but the label should be 1
        label[gt_max_iou_index] = 1
        label[max_iou>self.pos_thresh] = 1
        # select disable position ,then set the label to -1(ignore)

        pos_index = np.where(label==1)[0]
        neg_index = np.where(label==0)[0]
        n_pos = int(self.n_sample * self.pos_ratio)
        n_pos = min(n_pos, len(pos_index))
        n_neg = self.n_sample - n_pos
        n_neg = min(n_neg, len(neg_index))
        #
        disable_pos_index = np.random.choice(pos_index, len(pos_index) - n_pos, replace=False)
        disable_neg_index = np.random.choice(neg_index, len(neg_index) - n_neg, replace=False)
        label[disable_neg_index] = -1
        label[disable_pos_index] = -1
        # label set done
        # get loc
        # 注意这里直接计算所有的有效的anchor的loc即可，因为后续计算loss时，针对label=0和-1的情况会自动屏蔽
        loc = at.bbox2loc(anchor, gt_box[argmax_iou])
        # 映射回初始位置
        label = self._unmap(label, valid, n_anchor, -1)
        loc = self._unmap(loc, valid, n_anchor, 0)
        return loc, label


    def _unmap(self, x, index, n_anchor, fill):
        if len(x.shape)==1:
            res = np.empty((n_anchor,), dtype=x.dtype)
        else:
            res = np.empty((n_anchor, 4), dtype=x.dtype)
        res.fill(fill)
        res[index] = x
        return res



class Head_Label(object):

    def __init__(self, n_sample=128, pos_ratio=0.25, pos_thresh=0.5, neg_thresh=0.5,
                 loc_mean=(0., 0., 0., 0.), loc_std=(0.1, 0.1, 0.2, 0.2)):
        super(Head_Label, self).__init__()
        self.loc_mean = loc_mean
        self.loc_std = loc_std
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

    def __call__(self, roi, gt_box, gt_label):
        roi = at.toNumpy(roi)
        gt_box = at.toNumpy(gt_box)
        roi = np.concatenate((roi, gt_box), axis=0)
        iou = at.twobox_iou(roi, gt_box)
        max_iou = iou.max(axis=1)
        argmax_iou = iou.argmax(axis=1)
        #
        pos_index = np.where(max_iou>=self.pos_thresh)[0]
        neg_index = np.where(max_iou<self.neg_thresh)[0]

        n_pos = int(self.n_sample * self.pos_ratio)
        n_pos = min(n_pos, len(pos_index))
        n_neg = self.n_sample - n_pos
        n_neg = min(n_neg, len(neg_index))

        pos_index = np.random.choice(pos_index, n_pos, replace=False)
        neg_index = np.random.choice(neg_index, n_neg, replace=False)
        keep_index = np.append(pos_index, neg_index)
        label = gt_label[argmax_iou[keep_index]] + 1
        label[n_pos:] = 0
        sample_roi = roi[keep_index]
        sample_roi_gt = gt_box[argmax_iou[keep_index]]
        loc = at.bbox2loc(sample_roi, sample_roi_gt)
        loc = (loc - self.loc_mean) / self.loc_std
        return sample_roi, loc, label


