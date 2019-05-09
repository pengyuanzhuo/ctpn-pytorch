# coding: utf-8

class TrainConfig:
    TRAIN_DIR = '/Users/qiniu/Workspace/private/ctpn-pytorch/dataset/vis' # 训练集目录
    VAL_DIR = None # 验证集目录
    GPUS = '0' # GPU列表(例如0,1,2,3)
    CHECKPOINT_DIR = './checkpoints' # checkpoint dir
    CHECKPOINT = None # 从checkpoint 恢复训练
    PRINT_FREQ = 20

    FEAT_STRIDE = 16 # vgg16

    EPOCHS = 50
    LR = 0.001
    MOMENTUM = 0.9
    WD = 0.0005 # weight decay

    LAMBD_REG = 1.0 # bbox regression weight, 默认1.0