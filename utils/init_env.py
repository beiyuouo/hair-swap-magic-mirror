# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/22 23:55
# Description:

__author__ = "BeiYu"

import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True