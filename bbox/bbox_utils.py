# coding: utf-8

import numpy as np


def bbox_transform(bbox_dst, bbox_src):
    '''
    calc transform(bbox_src => bbox_dst)
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


def bbox_transform_inv(bbox_src, transform):
    '''
    calc bbox_dst = transform(bbox_src)
    Args:
        bbox_src: np.array, vstack [xmin, ymin, xmax, ymax]
        transform: np.array, vstack [dy, dh]

    Return:
        bbox_dst: np.array, vstack [xmin, ymin, xmax, ymax]
    '''
    assert bbox_src.shape[0] == transform.shape[0]

    h_src = bbox_src[:, 3] - bbox_src[:, 1] + 1.0
    src_bbox_centor_y = bbox_src[:, 1] + 0.5 * h_src

    dst_bbox_centor_y = src_bbox_centor_y + transform[:, 0]
    h_dst = h_src * np.exp(transform[:, 1])

    dst_y_min = dst_bbox_centor_y - 0.5 * h_dst
    dst_y_max = dst_bbox_centor_y + 0.5 * h_dst

    bbox_dst = np.vstack((bbox_src[:, 0],
                          dst_y_min,
                          bbox_src[:, 2],
                          dst_y_max)).transpose()

    return bbox_dst


def py_iou(bbox_a, bbox_b):
    '''
    Args:
        bbox_a: np.array, vstack [xmin, ymin, xmax, ymax], shape=(M, 4)
        bbox_b: np.array, vstack [xmin, ymin, xmax, ymax], shape=(N, 4)

    Return:
        overlaps: np.array, shape=(M, N),
            loc(i, j) means the iou between i'th box in bbox_a and j'th box in bbox_b
    '''
    if len(bbox_a.shape) == 1:
        bbox_a = bbox_a[np.newaxis, :]
    if len(bbox_b.shape) == 1:
        bbox_b = bbox_b[np.newaxis, :]
    xi1 = np.maximum(bbox_a[:, 0][:, np.newaxis], bbox_b[:, 0]) # (M, N)
    yi1 = np.maximum(bbox_a[:, 1][:, np.newaxis], bbox_b[:, 1]) # (M, N)
    xi2 = np.minimum(bbox_a[:, 2][:, np.newaxis], bbox_b[:, 2]) # (M, N)
    yi2 = np.minimum(bbox_a[:, 3][:, np.newaxis], bbox_b[:, 3]) # (M, N)
    inter_area = (xi2 - xi1 + 1) * (yi2 - yi1 + 1) # （M, N)

    bbox_a_area = (bbox_a[:, 2] - bbox_a[:, 0] + 1) * (bbox_a[:, 3] - bbox_a[:, 1] + 1) # (M, )
    bbox_b_area = (bbox_b[:, 2] - bbox_b[:, 0] + 1) * (bbox_b[:, 3] - bbox_b[:, 1] + 1) # (N, )

    union_area = bbox_a_area[:, np.newaxis] + bbox_b_area - inter_area

    overlaps = inter_area / union_area

    return overlaps


def py_nms(scores, bboxes, iou_theashold=0.5, max_boxes=None):
    '''
    Applies Non-max suppression (NMS) to set of boxes
    Python baseline
    Args:
        scores: np.array, shape=(N, )
        bboxes: np.array, shape=(N, 4), vstack [xmin, ymin, xmax, ymax]
        iou_theashold: real value, "intersection over union" threshold used for NMS filtering
        max_boxes: integer, maximum number of predicted boxes you'd like
    Return:
        scores: np.array, shape=(M, )
        bboxes: np.array, shape=(M, 4)
    '''
    indices = np.argsort(scores)
    indices = indices[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        i_ious = iou.py_iou(bboxes[i], bboxes[indices[1:]])
        i_ious = np.squeeze(i_ious)
        keep_indices = np.where(i_ious < iou_theashold)[0]
        indices = indices[keep_indices + 1]

    if max_boxes and len(keep) > max_boxes:
        keep = keep[:max_boxes]

    return scores[keep], bboxes[keep]


if __name__ == "__main__":
    src_bbox = np.array([[0, 0, 3, 3]])
    dst_bbox = np.array([[0.1, 0.1, 3.2, 3.0]])

