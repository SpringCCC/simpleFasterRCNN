# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 15:47
# @Author  : WeiHuang
import torch.nn as nn
from torchvision.ops import RoIPool
from models.head.config import opt
import torch.nn.functional as F
import utils.array_tool as at
import numpy as np

class VGG16Head(nn.Module):

    def __init__(self, classifier, n_class):
        super(VGG16Head, self).__init__()
        self.classifier = classifier
        self.n_class = n_class
        self.roi_pool = RoIPool(opt.output_size, opt.spatial_scale)
        self.head_loc_net = nn.Linear(opt.in_channel, (n_class+1)*4)
        self.head_score_net = nn.Linear(opt.in_channel, n_class+1)
        at.init_weight([self.head_score_net], 0, 0.01)
        at.init_weight([self.head_loc_net], 0, 0.001)

    def forward(self, x, roi):
        roi = at.toNumpy(roi)
        roi = at.cycleconvert_y1x1y2x2_x1y1x2y2(roi)
        indice = np.zeros((len(roi), 1), dtype=roi.dtype)
        roi = np.concatenate([indice, roi], axis=1)
        roi = at.toTensor(roi).float()
        roi_x = self.roi_pool(x, roi) # n, c, h, w
        roi_x = roi_x.reshape(roi_x.shape[0], -1)
        x = self.classifier(roi_x)
        head_loc = self.head_loc_net(x)
        head_score = self.head_score_net(x)
        return head_loc, head_score
