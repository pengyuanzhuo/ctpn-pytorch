# coding: utf-8

class Config:
    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7

    RPN_BATCHSIZE = 256 # 正负anchor数
    RPN_FG_FRACTION = 0.5 # 正样本比例

    MIN_SIZE = 16 # proposal的最小尺寸
    SCORE_THRESHOLD = 0.7 # > 该阈值的proposal被当作文本
    RPN_PRE_NMS_N = 12000 # nms前, 最多保留的box数
    RPN_POST_NMS_N = 2000 # nms后, 最多保留的box数
    NMS_THRESHOLD = 0.7 # nms阈值