# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 14:00
# @Author  : WeiHuang


import torch.nn as nn
import utils.array_tool as at
from models.rpn.config import opt
import torch.nn.functional as F
from models.rpn.proposalcreator import ProposalCreator

class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()
        self.base_anchor = at.get_base_anchor(opt.ratio, opt.scale, opt.feature_stride)
        n_base_anchor = len(self.base_anchor)
        self.conv1 = nn.Conv2d(opt.in_channel, opt.mid_channel, 3, 1, 1)
        self.rpn_loc_net = nn.Conv2d(opt.mid_channel, n_base_anchor * 4, 1, 1, 0)
        self.rpn_score_net = nn.Conv2d(opt.mid_channel, n_base_anchor * 2, 1, 1,0)
        at.init_weight([self.conv1, self.rpn_score_net, self.rpn_loc_net], 0, 0.01)
        self.proposalcreator = ProposalCreator()

    def forward(self, x, img_size):
        n, c, h, w = x.shape
        anchor = at.get_all_anchor(self.base_anchor, h, w, opt.feature_stride)
        h = self.conv1(x)
        rpn_loc = self.rpn_loc_net(h)[0] # chw
        rpn_score = self.rpn_score_net(h)[0] # chw
        rpn_loc = rpn_loc.permute(1,2, 0).reshape(-1, 4)
        rpn_score = rpn_score.permute(1,2, 0).reshape(-1, 2)
        rpn_score_softmax = F.softmax(rpn_score, dim=1)
        roi = self.proposalcreator(self, anchor, rpn_loc, rpn_score_softmax[:, 1], img_size)
        return roi, anchor, rpn_loc, rpn_score