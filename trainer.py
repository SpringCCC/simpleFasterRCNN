# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 16:28
# @Author  : WeiHuang



import torch.nn as nn
import torch

class Traninner(nn.Module):

    def __init__(self, model):
        super(Traninner, self).__init__()
        self.model = model


    def train_step(self, img, bbox, label):
        self.model.train()
        self.model.optimizer.zerd_grad()
        loss = self(img, bbox, label)
        loss.backward()
        self.model.optimizer.step()


    def forward(self):
        pass



    def loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        inweight = torch.zeros(gt_label.shape).cuda()
        inweight[gt_label>0] = 1
        inweight = inweight.reshape(-1, 1).repeat(1, 4)
        loss = self.smooth_l1(pred_loc, gt_loc, inweight, sigma)
        # 这里是所有正样本和负样本加起来的个数(label>=0)，忽略掉被忽略的样本(label=-1)
        # TODO 这里可以只要正样本吗？ 因为只有正样本才需要定位
        valid_num = (gt_label>=0).sum().float()
        return loss / valid_num

    def smooth_l1(self, pred, gt, inweight, sigma):
        delta = 1 / (sigma * sigma)
        diff = inweight * abs(pred - gt)
        flag = (diff < delta).float()
        y = flag*(0.5 / delta * diff * diff) + (1-flag)* (diff - 0.5*delta)
        return y.sum

