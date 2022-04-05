import torch
from torch.utils.data import DataLoader
from utils.config import opt
from data.dataset import Dataset, inverse_normalize
from tqdm import tqdm
from models.fastercnnvgg16 import FasterRCNNVGG16
from trainer import Traninner
import numpy as np
from utils.vis_tool import Visual

def train(**kwargs):
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


    opt._parse(kwargs)

    # dataset and dataloader
    trn_dataset = Dataset(opt)
    trn_dataloader = DataLoader(trn_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    # build model
    model = FasterRCNNVGG16()
    model = model.cuda()
    #
    trainer = Traninner(model)
    vistool = Visual(opt.env)
    for ii, (img, bbox, label, scale) in tqdm(enumerate(trn_dataloader)):
        img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
        trainer.train_step(img, bbox, label)
        #
        if ii%30==0:
            img_gt = inverse_normalize(img[0])
            vistool.vis_img_bboxs(img_gt, bbox[0], label[0], 'gt')
            bboxs, scores, labels = model.predict([img_gt], is_visual=True)
            vistool.vis_img_bboxs(img_gt, bboxs, labels, 'predict', scores)


if __name__ == '__main__':
    train()