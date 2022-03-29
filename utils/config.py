

class Config():


    # voc_data_dir = '/dataset/PASCAL2007/VOC2007/'
    voc_data_dir = "/fastdata/computervision/huangwei/data/train/fasterrcnn/VOCdevkit/VOC2007/"
    num_workers = 4
    min_size = 600
    max_size = 1000
    caffe_pretrain = False
    n_class = 20
    optim_name = 'SGD' # Adam
    lr = 0.01









    def _state_dic(self):
        state_dic = {k:v for k, v in self.__dict__.items() if not k.startswith("_")}
        return state_dic


    def _parse(self, kwargs):
        state_dic = self._state_dic()
        for k, v in kwargs.items():
            assert k in state_dic, "输入的键值:{} 有误".format(k)
            setattr(self, k, v)
        state_dic = self._state_dic()
        print(state_dic)



opt = Config()