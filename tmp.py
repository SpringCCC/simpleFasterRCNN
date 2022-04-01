# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 10:45
# @Author  : WeiHuang

import utils.array_tool as at
import numpy as np


x = np.asarray([[0,1,2, 3]])
x = at.toTensor(x)
y = x.repeat(2,2,1)
z = y.reshape(4,2,2)
print(z)
