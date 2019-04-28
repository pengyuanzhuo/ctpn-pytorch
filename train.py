# coding: utf-8

import torch
from dataset import dataset
from torchvision import transforms
from net.ctpn import CTPN
from net.loss import CtpnLoss
from rpn.anchor_target import anchor_target
from config import TrainConfig as cfg


def train_epoch(model, data_loader, criterion, optimizer):
    '''
    1 epoch
    '''
    for img, label in data_loader:
        # forward
        # score.shape = (1, 20, H, W)
        # loc.shape = (1, 20, H, W)
        # side_refinement.shape = (1, 10, H, W)
        score, loc, side_refinement = model(img)

        # target
        feat_map_size = (score.shape[-2], score.shape[-1])
        _, _, im_h, im_w = img.shape
        im_info = (im_h, im_w, None)

        score_targets, reg_targets = anchor_target(feat_map_size, label.numpy().squeeze(0), im_info, feat_stride=16)

        loss = criterion(score, loc, torch.from_numpy(score_targets), torch.from_numpy(reg_targets))

        print(loss)





if __name__ == '__main__':
    data_transforms = transforms.Compose([
        dataset.Rescale(short_side=600),
        dataset.ToTensor(),
        dataset.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    train_set = dataset.SplitBoxDataset(cfg.TRAIN_DIR, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, pin_memory=True)

    # model
    model = CTPN()
    # model.cuda()
    model.train()

    # loss
    criterion = CtpnLoss(cfg.LAMBD_REG)

    for epoch in range(cfg.EPOCHS):
        train_epoch(model, train_loader, criterion, None)
