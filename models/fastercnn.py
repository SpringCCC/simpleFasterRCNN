

import torch.nn as nn
from utils.config import opt
from torch.optim import SGD, Adam
from data.dataset import preprocess
import utils.array_tool as at
import numpy as np
import torch.nn.functional as F
from torchvision.ops import nms

class FasterRCNN(nn.Module):

    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.optimizer = self.get_optimizer()
        self.score_thresh = 0
        self.n_class = head.n_class

    def forward(self, x):
        n, c, h, w = x.shape
        img_size = (h, w)
        feature = self.extractor(x)
        roi = self.rpn(feature, img_size)
        head_loc, head_score = self.head(feature, roi)
        return roi, head_loc, head_score


    def predict(self, imgs, sizes=None, is_visual=False):
        self.eval()
        self.nms_thresh = 0.3
        img_h, img_w = imgs.shape[2:]
        prepare_imgs = []
        if is_visual:
            self.score_thresh = 0.7
            sizes = []
            for img in imgs:
                sizes.append(img.shape[1:])
                img = preprocess(at.toNumpy(img))
                prepare_imgs.append(img)
        else:
            self.score_thresh = 0.05
            prepare_imgs = imgs
        bboxs = []
        scores = []
        labels = []
        for img, size in zip(prepare_imgs, sizes):
            img = at.toTensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi, head_loc, head_score = self(img)
            roi = roi / scale # important
            n_head_loc = head_loc.shape[0]
            head_loc = head_loc.reshape(n_head_loc, -1, 4)
            loc_mean, loc_std = np.asarray(opt.loc_mean), np.asarray(opt.loc_std)
            loc_mean = at.toTensor(loc_mean).repeat(self.n_class+1)
            loc_std = at.toTensor(loc_std).repeat(self.n_class+1)
            head_loc = head_loc * loc_std[None] +loc_mean[None]
            head_loc = head_loc.reshape(-1, 4)
            roi = roi.reshape(n_head_loc, 1, 4).repeat(1,self.n_class+1, 1).reshape(-1, 4)
            roi = at.loc2bbox(at.toNumpy(head_loc), at.toNumpy(roi))
            roi[:, 0::2] = np.clip(roi[:, 0::2], 0, size[0])
            roi[:, 1::2] = np.clip(roi[:, 1::2], 0, size[1])

            head_score = F.softmax(at.toTensor(head_score), dim=1)
            predict_bboxs, predict_labels, predict_scores = self._suppress(roi, head_score)
            bboxs.append(predict_bboxs)
            scores.append(predict_scores)
            labels.append(predict_labels)
        return

    def _suppress(self, rois, scores):
        # roi:[N*21, 4], score:[N, 21]
        predict_bboxs = []
        predict_scores = []
        predict_labels = []
        n_box = len(scores)
        rois = rois.reshape(n_box, self.n_class, 4)
        for i in range(1, self.n_class):
            roi = rois[:, i, :]
            score = scores[:, i]
            roi = at.toNumpy(roi)
            score = at.toNumpy(score)
            keep = np.where(score > self.score_thresh)[0]
            # keep = score > self.score_thresh
            roi = roi[keep]
            score = score[keep]
            keep = nms(at.toTensor(roi), at.toTensor(score), self.nms_thresh)
            keep = at.toNumpy(keep)
            roi = roi[keep]
            score = score[keep]
            predict_bboxs.append(roi)
            predict_scores.append(score)
            label = np.ones((len(keep),))*(i-1)
            predict_labels.append(label)
        predict_bboxs = np.concatenate(predict_bboxs, axis=0).astype(np.float32)
        predict_labels = np.concatenate(predict_labels, axis=0).astype(np.float32)
        predict_scores = np.concatenate(predict_scores, axis=0).astype(np.float32)
        return predict_bboxs, predict_labels, predict_scores





    def get_optimizer(self):
        parameters = []
        for name, p in self.named_parameters():
            if p.requires_grad:
                if 'bias' in name:
                    parameters += [{'params':p, 'lr':opt.lr * 2, 'weight_decay':0.9}]
                else:
                    parameters += [{'params':p, 'lr':opt.lr, 'weight_decay':0.9}]

        if opt.optim_name == 'SGD':
            optimizer = SGD(parameters, momentum=0.9)
        elif opt.optim_name == 'Adam':
            optimizer = Adam(parameters)
        return optimizer

    def scale_lr(self, decay=0.9):
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] *= decay
        return self.optimizer
