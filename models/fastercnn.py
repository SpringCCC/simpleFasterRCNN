

import torch.nn as nn



class FasterRCNN(nn.Module):

    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head


    def forward(self, x):
        n, c, h, w = x.shape
        img_size = (h, w)
        feature = self.extractor(x)
        roi = self.rpn(feature, img_size)
        self.head(feature, roi)
