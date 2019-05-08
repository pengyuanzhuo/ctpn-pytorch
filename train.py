# coding: utf-8

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from dataset import dataset
from net.ctpn import CTPN
from net.loss import CtpnLoss
from rpn.anchor_target import anchor_target
from config import TrainConfig as cfg

gpus = cfg.GPUS
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
GPUS = list(map(lambda x: int(x) ,gpus.split(',')))
N_GPUS = torch.cuda.device_count()
USE_CUDA = True
if N_GPUS > 0:
    print('Using %d GPUs...' % N_GPUS)
else:
    print('FUCK, Using CPU...')
    USE_CUDA = False

CUR_DEVICE = torch.device('cuda:%s'%GPUS[0]) if USE_CUDA else torch.device('cpu')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.count = 0
        self.sum = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count


def train_epoch(model, data_loader, criterion, optimizer, epoch, use_cuda=True):
    '''
    1 epoch
    '''
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for i, (img, label) in enumerate(data_loader):
        # forward
        # score.shape = (1, 20, H, W)
        # loc.shape = (1, 20, H, W)
        # side_refinement.shape = (1, 10, H, W)
        if use_cuda:
            img = img.to(CUR_DEVICE)
        score, loc = model(img)

        # target
        feat_map_size = (score.shape[-2], score.shape[-1])
        _, _, im_h, im_w = img.shape
        im_info = (im_h, im_w, None)

        score_targets, reg_targets = anchor_target(feat_map_size, label.numpy().squeeze(0), im_info, feat_stride=16)
        score_targets = torch.from_numpy(score_targets).long().to(CUR_DEVICE)
        reg_targets = torch.from_numpy(reg_targets).to(CUR_DEVICE)

        optimizer.zero_grad()
        loss = criterion(score, loc, score_targets, reg_targets)
        losses.update(loss.item(), img.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print('Epoch {0} | {1}/{2}\t'
                  'Batch Time {batch_time.avg:.4f}({batch_time.val:.4f})\t'
                  'Loss {losses.avg:.4f}({losses.val:.4f})'.format(epoch, i, len(data_loader),
                    batch_time=batch_time, losses=losses))



if __name__ == '__main__':
    # torch.device()
    # torch.cuda.device(device) # context-manager that changes the selected device
    # torch.device()
    # torch.cuda.is_available() # returns a bool indicating if cuda is currently available
    # torch.cuda.device_count() # return the number of GPUs available
    # cuda context
    data_transforms = transforms.Compose([
        dataset.Rescale(short_side=600),
        dataset.ToTensor(),
        dataset.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    train_set = dataset.SplitBoxDataset(cfg.TRAIN_DIR, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True)

    # model
    model = CTPN()
    if N_GPUS > 1:
        model = nn.DataParallel(model)
    model.to(CUR_DEVICE)
    model.train()

    # loss
    criterion = CtpnLoss(cfg.LAMBD_REG)

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR,
                          momentum=cfg.MOMENTUM, weight_decay=cfg.WD)

    # lr_scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(cfg.EPOCHS):
        lr_scheduler.step()
        train_epoch(model, train_loader, criterion, optimizer, epoch, USE_CUDA)
