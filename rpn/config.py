# coding: utf-8

class Config:
    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7

    RPN_BATCHSIZE = 256 # 正负anchor数
    RPN_FG_FRACTION = 0.5 # 正样本比例