import torch
from torch.utils.data import DataLoader
from utils.config import opt
from data.dataset import Dataset
from tqdm import tqdm
from models.fastercnnvgg16 import FasterRCNNVGG16
from trainer import Traninner
import numpy as np


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
    for ii, (img, bbox, label, scale) in tqdm(enumerate(trn_dataloader)):
        img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
        trainer.train_step(img, bbox, label)


if __name__ == '__main__':
    train()