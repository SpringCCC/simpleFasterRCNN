import torch
from torch.utils.data import DataLoader
from utils.config import opt
from data.dataset import Dataset
from tqdm import tqdm
from models.fastercnnvgg16 import FasterRCNNVGG16


def train(**kwargs):
    opt._parse(kwargs)

    # dataset and dataloader
    trn_dataset = Dataset(opt)
    trn_dataloader = DataLoader(trn_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    # build model
    model = FasterRCNNVGG16()
    model = model.cuda()
    for img, bbox, label, scale in tqdm(trn_dataloader):
        img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
        model(img)


if __name__ == '__main__':
    train()