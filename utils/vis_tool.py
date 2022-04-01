# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 17:25
# @Author  : WeiHuang

from visdom import Visdom
from utils.config import opt
import torch
import cv2
import utils.array_tool as at
import numpy as np


VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)

class Visual(object):


    def __init__(self, env):
        self.vis = Visdom(env=env, use_incoming_socket=False)


    def reinit(self, env, **kwargs):
        self.vis = Visdom(env=env, use_incoming_socket=False, **kwargs)


    def vis_img(self, img, winname):
        img = self.adapt_size(img.copy())
        self.vis.image(img, win=winname, opts={'title':winname})

    def vis_imgs(self, imgs, winname):
        assert len(imgs.shape)==4
        self.vis.images(imgs, win=winname, opts={'title': winname})

    def vis_img_bboxs(self, img, bboxs, labels, scores, winname):
        #img: chw
        cv_img = np.transpose(img.copy(), axes=(1,2,0))
        for box, label, score in zip(bboxs, labels, scores):
            cv2.rectangle(cv_img, box[:2], box[2:])

    def adapt_size(self, img, max_size=1000):
        c, h, w = img.shape
        max_v = max(h, w)
        if max_v > max_size:
            scale = max_v / max_size
            neww, newh = int(w/scale), int(h/scale)
            newimg = np.transpose(img, (1, 2, 0))
            img = cv2.resize(newimg, (neww, newh))
            img = np.transpose(img, (2, 0, 1))
        return img


if __name__ == '__main__':
    img_path = r"/data/computervision/niuyanhao/数据质量/鱼眼/光照/光照异常/方城8场生长场_育肥_11单元_14栏位_20210506080454_0.jpg"
    img = cv2.imread(img_path)[:, :, ::-1] / 255
    img = np.transpose(img, (2, 0, 1))
    # imgs = np.random.random((4, 3, 100, 100))
    # imgs = at.toTensor(imgs)
    vis = Visual("sss")
    winname = "abced"
    # vis.vis_img(img, winname)
    vis.vis_img(img, winname)
