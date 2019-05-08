# coding: utf-8

import numpy as np
from rpn.generate_anchors import generate_anchors
from bbox.bbox_utils import bbox_transform_inv, py_nms
from rpn.config import Config as cfg


def bbox_clip(bboxes, h, w):
    '''
    clip boxes to image boundaries
    Args:
        bboxes: np.array, shape=(N, 4), vstack [xmin, ymin, xmax, ymax]
        h: int, img height
        w: int, img weight
    '''
    # xmin
    bboxes[:, [0]] = np.maximum(0, np.minumum(bboxes[:, [0]], w - 1))
    # ymin
    bboxes[:, [1]] = np.maximum(0, np.minumum(bboxes[:, [1]], h - 1))
    # xmax
    bboxes[:, [2]] = np.maximum(0, np.minumum(bboxes[:, [2]], w - 1))
    # ymax
    bboxes[:, [3]] = np.maximum(0, np.minumum(bboxes[:, [3]], h - 1))

    return bboxes


def bbox_filter(bboxes, min_size):
    '''
    rm bboxes edge < min_size
    Args:
        bboxes: (N, 4)
        min_size: 
    Return:
        keep indices
    '''
    ws = bboxes[:, 2] - bboxes[:, 0] + 1
    hs = bboxes[:, 3] - bboxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]

    return keep


def bbox_proposal(cls_map, reg_map, im_info, feat_stride=16):
    '''
    warp anchor to bbox
    Args:
        cls_map: score map, fg or bg output of RPN, shape=(N, 20, H, W), N=1
        reg_map: bbox regression map, dy and dh, shape=(N, 20, H, W), N=1
        im_info: tuple, (img_h, img_w, img_scale)
        feat_stride: downsampleing ratio,
            for vgg16, feat_stride=16

    Return:
        scores: proposal scores, shape=(K, ), K is the num of proposals after nms
        proposal: proposal bboxes, shape=(K, 4), vstack [xmin, ymin, xmax, ymax]

    '''
    img_h, img_w, img_scale = im_info

    ##################### step1, generate anchors ##############
    base_anchors = generate_anchors() # shape=(10, 4), vstack [xmin, ymin, xmax, ymax]
    N, _, feat_map_h, feat_map_w = cls_map.shape
    stride_x = np.arange(0, feat_map_w) * feat_stride
    stride_y = np.arange(0, feat_map_h) * feat_stride
    stride_x, stride_y = np.meshgrid(stride_x, stride_y)
    shifts = np.vstack((stride_x.ravel(), stride_y.ravel(),
                       stride_x.ravel(), stride_y.ravel())).transpose() # shape=(num_strides, 4)

    shifts = shifts[np.newaxis, :, :].transpose((1, 0, 2))
    all_anchors = base_anchors + shifts
    all_anchors = all_anchors.reshape((-1, 4)) # shape=(H*W, 4)
    num_all_anchors = all_anchors.shape[0]

    # reg_map 4D => 2D
    # bbox_deltas: vstack [ty, th]
    bbox_deltas = reg_map.reshape((N, 2, -1, feat_map_w)).transpose((0, 2, 3, 1)) # (N, 20, H, W) => (N, 10H, W, 2)
    bbox_deltas = bbox_deltas.reshape((-1, 2)) # (N*H*W*10, 2))

    # cls_map 4D => 2D
    scores = cls_map.transpose((0, 2, 3, 1)) # (N, 2A, H, W) => (N, H, W, 2A)
    scores = scores.reshape((N, H, W, n_anchors, 2)) # (N, H, W, 2A) => (N, H, W, A, 2)
    scores = scores[:, :, :, :, 1] # prob(fg)
    scores = scores.reshape((N, H, W, n_anchors)) # shape=(N, H, W, A)
    scores = scores.reshape((-1, 1)) # shape=(1*H*W*A, 1)
    scores = scores.ravel() # (H*W*A, )

    # 调整超出图像边界的proposal, 并去除<min_size的proposal
    proposal = bbox_transform_inv(all_anchors, bbox_deltas) # shape=(H*W, 4)
    proposal = bbox_clip(proposal, img_h, img_w)
    keep_indices = bbox_filter(proposal, cfg.MIN_SIZE)
    scores = scores[keep_indices]
    proposal = scores[keep_indices, :]

    # 筛选出score > score_threshold的proposal
    pos_indices = np.where(scores >= cfg.SCORE_THRESHOLD)[0]
    scores = scores[pos_indices]
    proposal = proposal[pos_indices]

    # 将scores和proposal按score的降序排序
    scores_indices = np.argsort(scores)[::-1]
    if scores_indices.size > cfg.RPN_PRE_NMS_N:
        scores_indices = scores_indices[:cfg.RPN_PRE_NMS_N]
    scores = scores[scores_indices]
    proposal = proposal[scores_indices, :]

    # nms
    scores, proposal = py_nms(scores, proposal, iou_threshold=cfg.NMS_THRESHOLD, max_boxes=cfg.RPN_POST_NMS_N)

    return scores, proposal


if __name__ == "__main__":
    bbox_proposal((4, 4))
