# coding: utf-8

class TrainConfig:
    TRAIN_DIR = '/Users/qiniu/Workspace/private/ctpn-pytorch/dataset/vis' # 训练集目录
    GPUS = '0' # GPU列表(例如0,1,2,3)
    EPOCHS = 50

    LAMBD_REG = 1.0 # bbox regression weight, 默认1.0