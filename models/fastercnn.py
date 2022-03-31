

import torch.nn as nn
from utils.config import opt
from torch.optim import SGD, Adam
from data.dataset import preprocess
import utils.array_tool as at
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
            n_head_loc = head_loc.shape[0]
            head_loc = head_loc.reshape(n_head_loc, -1, 4)
            loc_mean, loc_std = opt.loc_mean, opt.loc_std
            loc_mean = loc_mean.repeat(n_head_loc, self.n_class+1, 1)
            at.loc2bbox()



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
