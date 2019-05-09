# coding: utf-8

import os
import sys
import numpy as np
import traceback
from tqdm import tqdm
from shapely.geometry import Polygon
import cv2


def pickTopLeft(poly):
    idx = np.argsort(poly[:, 0])
    if poly[idx[0], 1] < poly[idx[1], 1]:
        s = idx[0]
    else:
        s = idx[1]

    return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]


def orderConvex(p):
    '''
    调整为顺时针顺序
    '''
    points = Polygon(p).convex_hull
    points = np.array(points.exterior.coords)[:4]
    points = points[::-1]
    points = pickTopLeft(points)
    points = np.array(points).reshape([4, 2])
    return points


def shrink_poly(poly, r=16):
    # y = kx + b
    x_min = int(np.min(poly[:, 0]))
    x_max = int(np.max(poly[:, 0]))

    k1 = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0])
    b1 = poly[0][1] - k1 * poly[0][0]

    k2 = (poly[2][1] - poly[3][1]) / (poly[2][0] - poly[3][0])
    b2 = poly[3][1] - k2 * poly[3][0]

    res = []

    start = int((x_min // 16 + 1) * 16)
    end = int((x_max // 16) * 16)

    p = x_min
    res.append([p, int(k1 * p + b1),
                start - 1, int(k1 * (p + 15) + b1),
                start - 1, int(k2 * (p + 15) + b2),
                p, int(k2 * p + b2)])

    for p in range(start, end + 1, r):
        res.append([p, int(k1 * p + b1),
                    (p + 15), int(k1 * (p + 15) + b1),
                    (p + 15), int(k2 * (p + 15) + b2),
                    p, int(k2 * p + b2)])
    return np.array(res, dtype=np.int).reshape([-1, 8])


if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    MAX_LEN = 1200
    MIN_LEN = 800

    im_fns = os.listdir(os.path.join(data_dir, "images"))
    im_fns.sort()

    if not os.path.exists(os.path.join(out_dir, "images")):
        os.makedirs(os.path.join(out_dir, "images"))
    if not os.path.exists(os.path.join(out_dir, "gt")):
        os.makedirs(os.path.join(out_dir, "gt"))

    # for each image
    for im_fn in tqdm(im_fns):
        try:
            _, fn = os.path.split(im_fn)
            bfn, ext = os.path.splitext(fn)
            if ext.lower() not in ['.jpg', '.png']:
                continue

            gt_path = os.path.join(data_dir, "gt", bfn + '.txt')
            img_path = os.path.join(data_dir, "images", im_fn)

            img = cv2.imread(img_path)
            img_size = img.shape
            im_size_min = np.min(img_size[0:2])
            im_size_max = np.max(img_size[0:2])

            im_scale = float(MIN_LEN) / float(im_size_min)
            if np.round(im_scale * im_size_max) > MAX_LEN:
                im_scale = float(MAX_LEN) / float(im_size_max)
            new_h = int(img_size[0] * im_scale)
            new_w = int(img_size[1] * im_scale)

            new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16 # 图片扩一个anchor的宽度
            new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

            re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            re_size = re_im.shape

            # 一张图片中的所有polys
            polys = []
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                splitted_line = line.strip().strip('\ufeff').lower().split(',')
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, splitted_line[:8])
                poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])

                poly[:, 0] = poly[:, 0] / img_size[1] * re_size[1]
                poly[:, 1] = poly[:, 1] / img_size[0] * re_size[0]

                poly = orderConvex(poly)
                polys.append(poly)

                # cv.polylines(re_im, [poly.astype(np.int32).reshape((-1, 1, 2))], True,color=(0, 255, 0), thickness=2)

            res_polys = []
            for poly in polys:
                # delete polys with width less than 10 pixel
                if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
                    continue

                res = shrink_poly(poly) # 每个poly可以分成多个小区域
                # for p in res:
                #    cv2.polylines(re_im, [p.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

                res = res.reshape([-1, 4, 2])
                # 处理每个小区域, 将不规则四边形转成矩形(xmin, ymin, x_max, y_max)
                for r in res:
                    x_min = np.min(r[:, 0])
                    y_min = np.min(r[:, 1])
                    x_max = np.max(r[:, 0])
                    y_max = np.max(r[:, 1])

                    res_polys.append([x_min, y_min, x_max, y_max])

            cv2.imwrite(os.path.join(out_dir, "images", fn), re_im)
            with open(os.path.join(out_dir, "gt", bfn) + ".txt", "w") as f:
                for p in res_polys:
                    line = ",".join(str(p[i]) for i in range(4))
                    f.writelines(line + "\r\n")
                    # for p in res_polys:
                    #    cv2.rectangle(re_im,(p[0],p[1]),(p[2],p[3]),color=(0,0,255),thickness=1)

                    #cv2.imshow("demo",re_im)
                    #cv2.waitKey(0)
        except Exception as e:
            traceback.print_exc()
            print("Error processing {}".format(im_fn))

