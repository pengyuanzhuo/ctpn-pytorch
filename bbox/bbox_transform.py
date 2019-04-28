# coding: utf-8

import numpy as np


def bbox_transform(bbox_dst, bbox_src):
    '''
    transform(bbox_src => bbox_dst)
    Args:
    -----
    bbox_dst: np.array, N vstack [xmin, ymin, xmax, ymax]
    bbox_src: np.array, N vstack [xmin, ymin, xmax, ymax]

    Return:
    ------
    targets: vstack N [dx, dy, dw, dh], shape=(N, 4)
    '''
    assert bbox_src.shape == bbox_dst.shape

    h_src = bbox_src[:, 3] - bbox_src[:, 1] + 1.0 # shape=(N, )
    w_src = bbox_src[:, 2] - bbox_src[:, 0] + 1.0
    src_bbox_centor_y = bbox_src[:, 1] + 0.5 * h_src
    src_bbox_centor_x = bbox_src[:, 0] + 0.5 * w_src

    h_dst = bbox_dst[:, 3] - bbox_dst[:, 1] + 1.0
    w_dst = bbox_dst[:, 2] - bbox_dst[:, 0] + 1.0
    dst_bbox_centor_y = bbox_dst[:, 1] + 0.5 * h_dst
    dst_bbox_centor_x = bbox_dst[:, 0] + 0.5 * w_dst

    dx = (dst_bbox_centor_x - src_bbox_centor_x) / w_src
    dy = (dst_bbox_centor_y - src_bbox_centor_y) / h_src
    dh = np.log(h_dst / h_src)
    dw = np.log(w_dst / w_src)

    # targets = np.vstack((dx, dy, dh, dw)).transpose()
    targets = np.vstack((dy, dh)).transpose()

    return targets


def bbox_transform_inv():
    pass
