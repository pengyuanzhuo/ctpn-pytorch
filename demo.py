# coding: utf-8

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
    img = np.ascontiguousarray(img[:, :, ::-1])
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    img -= mean[:, None, None]
    img /= std[:, None, None]

    return img[None, :, :, :]


def infer(input, model):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str, help='image path')
    parser.add_argument('model', type=str, help='model path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    img = cv2.imread(args.img)
    inputs = preprocess(img, 768)

    print(inputs.shape)

    # load model
    # model = TODO
    # model.eval()

    # out = infer(inputs, model)

    # draw
    # TODO
