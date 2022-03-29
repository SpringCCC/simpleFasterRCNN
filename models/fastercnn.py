

import torch.nn as nn
from utils.config import opt
from torch.optim import SGD, Adam

class FasterRCNN(nn.Module):

    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.optimizer = self.get_optimizer()


    def forward(self, x):
        n, c, h, w = x.shape
        img_size = (h, w)
        feature = self.extractor(x)
        roi = self.rpn(feature, img_size)
        self.head(feature, roi)

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
