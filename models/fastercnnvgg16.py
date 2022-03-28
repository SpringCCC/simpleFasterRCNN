

import torch.nn as nn
from models.extractor.decom import decomvgg16
from models.rpn.region_proposal_net import RPN
from models.fastercnn import FasterRCNN
from models.head.vgghead import VGG16Head
from utils.config import opt


class FasterRCNNVGG16(FasterRCNN):

    def __init__(self):

        extractor, classifier = decomvgg16()
        rpn = RPN()
        head = VGG16Head(classifier, opt.n_class)
        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)