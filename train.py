# coding: utf-8

import os
import time
import shutil
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

best_loss = float('inf')


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
    tic = time.time()

    model.train()
    for i, (img, label, img_info) in enumerate(data_loader):
        # forward
        # score.shape = (1, 20, H, W)
        # loc.shape = (1, 20, H, W)
        # side_refinement.shape = (1, 10, H, W)
        if use_cuda:
            img = img.to(CUR_DEVICE)
        score, loc = model(img)

        # target
        feat_map_size = (score.shape[-2], score.shape[-1])
        img_info = list(map(lambda x: x.item(), img_info))

        score_targets, reg_targets = anchor_target(feat_map_size, label.numpy().squeeze(0),
                                                   img_info, feat_stride=cfg.FEAT_STRIDE)
        score_targets = torch.from_numpy(score_targets).long().to(CUR_DEVICE)
        reg_targets = torch.from_numpy(reg_targets).to(CUR_DEVICE)

        optimizer.zero_grad()
        loss = criterion(score, loc, score_targets, reg_targets)
        losses.update(loss.item(), img.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % cfg.PRINT_FREQ == 0:
            print('Epoch {0} | {1}/{2}\t'
                  'Batch Time {batch_time.avg:.4f}({batch_time.val:.4f})\t'
                  'Loss {losses.avg:.4f}({losses.val:.4f})'.format(epoch, i, len(data_loader),
                    batch_time=batch_time, losses=losses))

        return losses.avg


def validate(model, val_loader, criterion, use_cuda=True):
    print('val...')
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for i, (img, label, img_info) in enumerate(val_loader):
            if use_cuda:
                img = img.to(CUR_DEVICE, non_blocking=True)
            score, loc = model(img)

            feat_map_size = (score.shape[-2], score.shape[-1])
            img_info = list(map(lambda x: x.item(), img_info))

            score_targets, reg_targets = anchor_target(feat_map_size, label.numpy().squeeze(0),
                                                       img_info, feat_stride=cfg.FEAT_STRIDE)
            score_targets = torch.from_numpy(score_targets).long().to(CUR_DEVICE)
            reg_targets = torch.from_numpy(reg_targets).to(CUR_DEVICE)

            loss = criterion(score, loc, score_targets, reg_targets)
            losses.update(loss.item(), img.size(0))

    return losses.avg


def main():
    global best_loss
    start_epoch = 0

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    data_transforms = transforms.Compose([
        dataset.Rescale(short_side=600),
        dataset.ToTensor(),
        dataset.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    train_set = dataset.SplitBoxDataset(cfg.TRAIN_DIR, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                               shuffle=True, pin_memory=True)

    if cfg.VAL_DIR is not None:
        val_set = dataset.SplitBoxDataset(cfg.VAL_DIR, transform=data_transforms)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,
                                                 shuffle=True, pin_memory=True)

    model = CTPN()
    if N_GPUS > 1:
        model = nn.DataParallel(model)
    model.to(CUR_DEVICE)

    criterion = CtpnLoss(cfg.LAMBD_REG)
    optimizer = optim.SGD(model.parameters(), lr=cfg.LR,
                          momentum=cfg.MOMENTUM, weight_decay=cfg.WD)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if cfg.CHECKPOINT:
        checkpoint = torch.load(cfg.CHECKPOINT)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_state_dict'])
        print('=> loaded checkpoint {} (epoch {})'.format(cfg.CHECKPOINT, start_epoch))

    for epoch in range(start_epoch, cfg.EPOCHS):
        lr_scheduler.step()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, USE_CUDA)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'lr_state_dict': lr_scheduler.state_dict(),
            'loss': train_loss
        }
        save_path = os.path.join(cfg.CHECKPOINT_DIR, 'ctpn_epoch_{}.pth.tar'.format(epoch))
        torch.save(checkpoint, save_path)

        if cfg.VAL_DIR is None:
            continue

        val_loss = validate(model, val_loader, criterion, USE_CUDA)
        if val_loss > best_loss:
            print('best epoch: {}, train loss: {.3f}, val loss: {.3f}'.format(epoch, train_loss, val_loss))
            print('saving checkpoint to {}...'.format(cfg.CHECKPOINT_DIR))
            best_checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, 'checkpoint_best.pth.tar')
            shutil.copy2(save_path, best_checkpoint_path)


if __name__ == '__main__':
    # torch.device()
    # torch.cuda.device(device) # context-manager that changes the selected device
    # torch.device()
    # torch.cuda.is_available() # returns a bool indicating if cuda is currently available
    # torch.cuda.device_count() # return the number of GPUs available
    # cuda context
    main()
