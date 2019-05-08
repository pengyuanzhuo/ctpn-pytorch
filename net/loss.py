# coding: utf-8

import torch
import torch.nn as nn


class CtpnLoss(nn.Module):
    def __init__(self, lambd_reg=1.0):
        super(CtpnLoss, self).__init__()
        self.lambd_reg = lambd_reg
        # self.lambd_refinement = 2.0
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()

    def forward(self, score, loc, score_target, loc_target):
        '''
        :param score: prediction score, shape=(N, 10*2, H, W)
        :param loc: prediction loc, shape=(N, 10*2, H, W)
        :param side_refinement:
        :param score_target: gt anchor label from rpn target. shape=(N, 10, H, W)
        :param loc_target: gt bbox regression target from rpn target. shape=(N, 20, H, W)
        :param side_refinement_target
        :return total loss
        '''
        # part1, classification loss
        score = score.reshape((score.shape[0], 2, -1, score.shape[-1])) # shape=(N, 2, 10H, W)
        score = score.permute((0, 2, 3, 1)) # shape=(N, 10H, W, 2)
        score = score.reshape((-1, 2)) # shape=(num_all_anchor, 2)

        score_target = score_target.reshape((score_target.shape[0], 1, -1, score_target.shape[-1])) # shape=(1, 1, 10H, W)
        score_target = score_target.permute((0, 2, 3, 1)) # (1, 10H, W, 1)
        score_target = score_target.reshape((-1))

        valid_indices = (score_target >= 0).nonzero()[:, 0] # positive & negative label indices
        # score = score[valid_indices, :]
        # score_target = score_target[valid_indices]

        cls_loss = self.CrossEntropyLoss(score, score_target)

        # part2, bbox regression loss
        # normalized by (num of valid_samples)
        # DIFFRENR FROM PAPER !!!
        # paper: valid sample(positive or iou > 0.5 anchors)
        # code: valid sample(positive anchors)
        loc = loc.reshape((loc.shape[0], 2, -1, loc.shape[-1])) # (N, 2, 10H, W)
        loc = loc.permute((0, 2, 3, 1)) # (N, 2, 10H, W) => (N, 10H, W, 2)
        loc = loc.reshape((-1, 2)) # (N*10H*W, 2)
        loc = loc[valid_indices, :]

        # loc_target = loc_target[:, (1, 3), :, :] # shape=(1, 20, H, W)
        loc_target = loc_target.reshape((loc_target.shape[0], 2, -1, loc_target.shape[-1])) # shape=(N, 2, 10H, W)
        loc_target = loc_target.permute((0, 2, 3, 1)) # shape=(N, 10H, W, 2)
        loc_target = loc_target.reshape((-1, 2))
        loc_target = loc_target[valid_indices, :]

        assert loc.shape == loc_target.shape

        loc_loss = self.SmoothL1Loss(loc, loc_target)

        # part3, side_refinement loss
        # TODO

        return cls_loss + self.lambd_reg * loc_loss

if __name__ == '__main__':
    loss = CtpnLoss()
    score = torch.rand((1, 20, 32, 32))
    score_target = torch.rand((1, 10, 32, 32))
    score_target = (score_target > 0.5).long()
    loc = torch.rand((1, 20, 32, 32))
    loc_target = torch.rand((1, 20, 32, 32))
    ls = loss(score, loc, score_target, loc_target)
    print(ls)

