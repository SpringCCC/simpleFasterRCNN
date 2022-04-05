# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 14:01
# @Author  : WeiHuang

import torch.nn as nn
from models.rpn.config import opt
import utils.array_tool as at
import numpy as np
from torchvision.ops import nms

class ProposalCreator(nn.Module):
    def __init__(self):
        super(ProposalCreator, self).__init__()

    def forward(self, parent, anchor, loc, score, img_size):
        if parent.training:
            n_pre_nms = opt.n_train_pre_nms
            n_post_nms = opt.n_train_post_nms
        else:
            n_pre_nms = opt.n_test_pre_nms
            n_post_nms = opt.n_test_post_nms
        anchor = at.toNumpy(anchor)
        loc = at.toNumpy(loc)
        score = at.toNumpy(score)
        h, w = img_size
        anchor = at.loc2bbox(loc, anchor)
        anchor[:, 0::2] = np.clip(anchor[:, 0::2], 0, h)
        anchor[:, 1::2] = np.clip(anchor[:, 1::2], 0, w)
        #clean short anchor
        h,w = anchor[:, 2] - anchor[:, 0], anchor[:, 3] - anchor[:, 1]
        keep = np.where((h >= opt.feature_stride) & (w >= opt.feature_stride))[0]
        anchor = anchor[keep]
        score = score[keep]
        index = score.argsort()[::-1][:n_pre_nms]
        anchor = anchor[index]
        score = score[index]
        # nms
        anchor_nms = at.cycleconvert_y1x1y2x2_x1y1x2y2(anchor)
        keep = nms(at.toTensor(anchor_nms), at.toTensor(score), opt.nms_thresh)
        keep = keep[:n_post_nms]
        roi = anchor[at.toNumpy(keep)]
        return roi



