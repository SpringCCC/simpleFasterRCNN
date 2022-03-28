# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 13:35
# @Author  : WeiHuang

from torchvision.models import vgg16
from models.extractor.config import opt
import torch.nn as nn

def decomvgg16():
    model = vgg16(pretrained=True)
    features = list(model.features)[:30]
    classifier = list(model.classifier)[:6]
    if not opt.use_drop:
        classifier.pop(5)
        classifier.pop(2)
    # freeze top 10 layer
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    return nn.Sequential(*features), nn.Sequential(*classifier)



if __name__ == '__main__':
    decomvgg16()
