# coding: utf-8

from __future__ import print_function, division
import os
import csv
import cv2
import glob
import random
import numpy as np
import torch
import torch.utils.data as data


def get_images(img_dir):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(img_dir, '*.{}'.format(ext))))
    return files


def load_annoataion(gt_file):
    '''
    load annotation from the text file
    :param gt_file:
    :return: np.array, shape=(N, 4)
    '''
    text_polys = []
    with open(gt_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            xmin, ymin, xmax, ymax = list(map(float, line[:8]))
            text_polys.append([xmin, ymin, xmax, ymax])
    return np.array(text_polys, dtype=np.float32)


class SplitBoxDataset(data.Dataset):
    """ctpn dataset."""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): dataset path with subdir 'images' and 'gt'
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.img_dir = os.path.join(data_dir, 'images')
        self.img_fn_list = get_images(self.img_dir)
        self.n = len(self.img_fn_list)
        print('=> %d images in %s' % (self.n, data_dir))
        self.gt_dir = os.path.join(data_dir, 'gt')
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img_fn = self.img_fn_list[idx]
        _, img_name = os.path.split(img_fn)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_fn = os.path.join(self.gt_dir, txt_name)
        if not os.path.exists(txt_fn):
            print('gt file: %s not exist' % txt_fn)
            return self.__getitem__(random.randint(0, self.n - 1))
        gt_bboxes = load_annoataion(txt_fn)

        img = cv2.imread(img_fn)
        if img is None or gt_bboxes.shape[0] == 0:
            return self.__getitem__(random.randint(0, self.n - 1))

        sample = (img, gt_bboxes)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    def __init__(self, short_side=600):
        self.short_side = short_side

    def __call__(self, sample):
        img, gt_bboxes = sample
        h, w, _ = img.shape
        scale = float(self.short_side) / float(min(h, w))
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        h_scale = float(img.shape[0]) / h
        w_scale = float(img.shape[1]) / w
        gt_bboxes[:, (0, 2)] = gt_bboxes[:, (0, 2)] * w_scale
        gt_bboxes[:, (1, 3)] = gt_bboxes[:, (1, 3)] * h_scale

        return (img, gt_bboxes)


class ToTensor(object):
    def __call__(self, sample):
        img, gt_bboxes = sample
        img = np.ascontiguousarray(img[:, :, ::-1])
        img = img.transpose((2, 0, 1))
        return (torch.from_numpy(img).float(), torch.from_numpy(gt_bboxes))


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        img, gt_bboxes = sample
        if not (torch.is_tensor(img) and img.ndimension()) == 3:
            raise TypeError('img is not a torch image.')

        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        img -= mean[:, None, None]
        img /= std[:, None, None]

        return (img, gt_bboxes)


if __name__ == "__main__":
    pass
