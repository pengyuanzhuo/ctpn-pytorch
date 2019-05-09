# coding: utf-8

import os
import argparse
import cv2
import numpy as np
from rpn.proposal import bbox_proposal
from net.ctpn import CTPN

import torch


def preprocess(img, im_size):
    h, w, _ = img.shape
    scale = float(im_size) / float(min(h, w))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    h_new, w_new, _ = img.shape
    img = np.ascontiguousarray(img[:, :, ::-1])
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    img -= mean[:, None, None]
    img /= std[:, None, None]

    im_info = (h_new, w_new, scale)

    return img[None, :, :, :], im_info


def infer(inputs, im_info, model):
    model.eval()
    with torch.no_grad():
        cls_map, reg_map = model(inputs)
        scores, proposal = bbox_proposal(cls_map.numpy(), reg_map.numpy(),
                                         im_info, feat_stride=16)

    return scores, proposal


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str, help='image path')
    parser.add_argument('checkpoint', type=str, help='checkpoint path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    img = cv2.imread(args.img)
    inputs, im_info = preprocess(img, 768)

    checkpoint = torch.load(args.checkpoint)
    model = CTPN()
    model.load_state_dict(checkpoint['model_state_dict'])

    scores, proposal = infer(inputs, im_info, model)
    print(scores.shape)
    print(proposal.shape)

    img = cv2.imread(args.img)
    h, w, _ = img.shape
    scale = float(768) / float(min(h, w))
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    for i in proposal.shape[0]:
        bbox = proposal[i]
        poly = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
        cv2.polylines(img, [poly], True, color=(255,255,0), thickness=3)
    cv2.imwrite('./out.jpg', img)
