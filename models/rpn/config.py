

class Config():
    ratio = [0.5, 1, 2]
    scale = [8, 16, 32]
    n_train_pre_nms = 12000
    n_train_post_nms = 2000
    n_test_pre_nms = 2000
    n_test_post_nms = 300
    feature_stride = 16
    in_channel = 512
    mid_channel = 512
    nms_thresh = 0.7



opt = Config()