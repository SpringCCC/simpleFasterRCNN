# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 16:28
# @Author  : WeiHuang



import torch.nn as nn
import torch
from models.label_assign_tool import RPN_Label, Head_Label
import torch.nn.functional as F
from utils.config import opt
import utils.array_tool as at
class Traninner(nn.Module):

    def __init__(self, model):
        super(Traninner, self).__init__()
        self.model = model
        self.head_label = Head_Label(loc_mean=opt.loc_mean, loc_std=opt.loc_std)
        self.rpn_label = RPN_Label()


    def train_step(self, img, bbox, label):
        self.model.train()
        self.model.optimizer.zero_grad()
        loss = self(img, bbox, label)
        loss.backward()
        self.model.optimizer.step()


    def forward(self, img, gt_bbox, gt_label):
        n, c, h, w = img.shape
        img_size = (h, w)
        gt_label = gt_label[0]
        gt_bbox = gt_bbox[0]
        feature = self.model.extractor(img)
        roi, anchor, rpn_loc, rpn_score = self.model.rpn(feature, img_size)
        sample_roi, head_gt_loc, head_gt_label =self.head_label(roi, gt_bbox, gt_label)
        head_loc, head_score = self.model.head(feature, sample_roi)
        rpn_gt_loc, rpn_gt_label = self.rpn_label(anchor, gt_bbox, img_size)
        # rpn loss
        rpn_cls_loss = F.cross_entropy(at.toTensor(rpn_score), at.toTensor(rpn_gt_label).long(), ignore_index=-1)
        rpn_loc_loss = self.loc_loss(at.toTensor(rpn_loc), at.toTensor(rpn_gt_loc), at.toTensor(rpn_gt_label).long(), opt.rpn_sigma)
        # head loss
        head_cls_loss = F.cross_entropy(at.toTensor(head_score), at.toTensor(head_gt_label).long(), ignore_index=-1)
        # head_loc:n*84
        n_head_loc = head_loc.shape[0]
        head_loc = head_loc.reshape((n_head_loc, -1, 4))
        # head_loc:n*21
        head_predict_label = head_score.argmax(dim=1)
        head_loc = head_loc[torch.arange(0,n_head_loc).long().cuda(), head_predict_label]
        head_loc_loss = self.loc_loss(at.toTensor(head_loc), at.toTensor(head_gt_loc),
                                      at.toTensor(head_gt_label).long(), opt.head_sigma)
        loss = rpn_cls_loss + rpn_loc_loss + head_cls_loss + head_loc_loss
        return loss


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
        return y.sum()

