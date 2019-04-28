# coding: utf-8

import numpy as np


def scale_anchor(base_anchor, h, w):
    '''
    adjust base_anchor
    Args:
    ----
    base_anchor: [xmin, ymin, xmax, ymax]
    h: target anchor height
    w: target anchor width
    Return:
    ------
    anchor: [xmin, ymin, xmax, ymax], anchor form base_anchor
    '''
    x_centor = (base_anchor[0] + base_anchor[2]) / 2.
    y_centor = (base_anchor[1] + base_anchor[3]) / 2.

    xmin = x_centor - w / 2
    xmax = x_centor + w / 2
    ymin = x_centor - h / 2
    ymax = x_centor + h / 2

    return np.array([xmin, ymin, xmax, ymax], dtype=np.int32)


def generate_anchors(ws=[16], hs=[11, 16, 23, 33, 48, 68, 97, 139, 198, 283], base_size=16):
    '''
    '''
    base_anchor = np.array([0, 0, base_size-1, base_size-1], dtype=np.int32)
    anchors = []
    for w in ws:
        for h in hs:
            anchors.append(scale_anchor(base_anchor, h, w))

    return np.vstack(anchors)


if __name__ == '__main__':
    import time

    tic = time.time()
    a = generate_anchors()
    print(time.time() - tic)
    print(a)

