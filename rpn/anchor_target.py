# coding: utf-8

import numpy as np
from rpn.generate_anchors import generate_anchors
from rpn.config import Config as cfg
from bbox.bbox_utils import bbox_transform
from bbox.bbox_utils import py_iou


def anchor_target(feat_map_size, gt_boxes, im_info, feat_stride=16):
    '''
    Args:
    ----
        feat_map_size: (H, W)
        gt_boxes: shape=(N, 4) [xmin, ymin, xmax, ymax]
        im_info: [img_height, img_width, scale_ratio] ori img to net input img
        feat_stride: downsampleing ratio, for vgg16, feat_strides=16

    Return:
    ------
    labels: anchor labels, shape=(1, 10, h, w)
    reg_targets: bbox regression target, shape=(1, 40, h, w,)
    '''
    ############################ prepare anchors #########################
    # step1, origin anchors
    anchors = generate_anchors() # shape=(num_anchors, 4)
    _num_anchors = anchors.shape[0] # 10 anchors

    # step2, stride anchors on image
    feat_map_h, feat_map_w = feat_map_size
    shift_x = np.arange(0, feat_map_w) * feat_stride
    shift_y = np.arange(0, feat_map_h) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose() # shape=(num_strides, 4)

    anchors = anchors[None, :, :] # new axis, shape=(1, num_anchors, 4)
    shifts = shifts[None, :, :] # new axis, shape=(1, num_strides, 4)
    shifts = shifts.transpose((1, 0, 2)) # shape=(num_strides, 1, 4)

    all_anchors = anchors + shifts # shape=(num_strides, num_anchors, 4)
    all_anchors = all_anchors.reshape((-1, 4))
    num_all_anchors = all_anchors.shape[0]

    # step3, rm outside anchors
    im_h, im_w, _ = im_info
    border = 0
    inside_indices = np.where((anchors[:, 0] >= -border) & # bool list
                              (anchors[:, 1] >= -border) &
                              (anchors[:, 2] < im_w + border) &
                              (anchors[:, 3] < im_h + border))[0]

    anchors = all_anchors[inside_indices, :]
    num_anchors = len(inside_indices)

    ############################ anchor label ###########################
    # positive 1
    # negative 0
    # ignore -1
    labels = np.empty((len(inside_indices), ), dtype=np.float32)
    labels.fill(-1)

    iou_mat = py_iou(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    ) # shape=(num_anchors, num_gts)

    # max gt for anchors
    maxgt_indices = iou_mat.argmax(axis=1)
    maxgt_iou = iou_mat[np.arange(num_anchors), maxgt_indices] # shape=(num_anchors, )

    labels[maxgt_iou < cfg.RPN_NEGATIVE_OVERLAP] = 0
    labels[maxgt_iou >= cfg.RPN_POSITIVE_OVERLAP] = 1

    # max anchor for gts
    maxanchor_indices = iou_mat.argmax(axis=0)
    labels[maxanchor_indices] = 1

    # positive num <= const
    max_potitive_num = cfg.RPN_BATCHSIZE * cfg.RPN_FG_FRACTION
    positive_num = len(np.where(labels == 1)[0])
    if positive_num > max_potitive_num:
        drop_indices = np.random.choice(np.where(label == 1)[0],
                                        max_potitive_num - positive_num,
                                        replace=False)
        labels[drop_indices] = -1

    # negative num + positive num == RPN_BATCHSIZE
    max_negative_num = cfg.RPN_BATCHSIZE - np.sum(labels == 1)
    negative_indices = np.where(labels == 0)[0]
    negative_num = len(negative_indices)
    if negative_num > max_negative_num:
        drop_indices = np.random.choice(negative_indices,
                                        negative_num - max_negative_num,
                                        replace=False)
        labels[drop_indices] = -1

    ############################# rpn bbox ############################
    reg_targets = np.empty((len(inside_indices), 4), dtype=np.float32)
    reg_targets = bbox_transform(anchors, gt_boxes[maxgt_indices, :]) # shape=(num_anchors, 4)

    labels = _unmap(labels, all_anchors.shape[0], inside_indices, -1) # shape=(1, num_all_anchors)
    reg_targets = _unmap(reg_targets, all_anchors.shape[0], inside_indices, 0) # shape=(num_all_anchors, 4)

    labels = labels.reshape((1, feat_map_h, feat_map_w, _num_anchors)).transpose((0, 3, 1, 2))
    reg_targets = reg_targets.reshape((1, feat_map_h, feat_map_w, _num_anchors*2)).transpose((0, 3, 1, 2))

    return labels, reg_targets


def _unmap(data, count, inds, fill=0):
    '''
    unmap a subset of item(data) back to the original set of items(size=count)
    '''
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data

    return ret


if __name__ == '__main__':
    pass

