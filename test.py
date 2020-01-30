#!/usr/bin/env python
# coding: utf-8
# File: test.py

# This file is used for testing.

from func import *
from sync_bn import *
from parallel import *

from torch import nn

import os
import torch

DEV_IDS = [1, 5]
DEV = torch.device('cuda:{}'.format(DEV_IDS[0]) if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(dev_id) for dev_id in DEV_IDS])


class ModelBn(nn.Module):
    def __init__(self, num_features=3):
        super(ModelBn, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, inputs):
        outputs = self.bn(inputs)
        return outputs


class ModelSynBn(nn.Module):
    def __init__(self, num_features=3):
        super(ModelSynBn, self).__init__()
        self.bn = SynchronizedBatchNorm2d(num_features=num_features)

    def forward(self, inputs):
        outputs = self.bn(inputs)
        return outputs


if __name__ == '__main__':
    model_syn_bn = ModelSynBn()
    model_syn_bn = DataParallelWithCallBack(model_syn_bn, device_ids=DEV_IDS)
    model_syn_bn.to(DEV)
    print(model_syn_bn)

    x = torch.randint(low=0, high=256, size=(4, 3, 256, 256), device=DEV).float()
    print(x)

    y = model_syn_bn(x)
    print(y)

    # *mean*
    print(y.mean(dim=(0, 2, 3)))
    # *std*
    print(y.std(dim=(0, 2, 3)))

    model_bn = ModelBn()
    model_bn = nn.DataParallel(model_bn, device_ids=DEV_IDS)
    model_bn.to(DEV)
    print(model_bn)

    y = model_bn(x)
    print(y)

    # *mean*
    print(y.mean(dim=(0, 2, 3)))
    # *std*
    print(y.std(dim=(0, 2, 3)))

    # *Use 'convert_model' to onvert input module and its child recursively*
    model_cvt = convert_model(model_bn)
    print(model_cvt)

    y = model_cvt(x)
    print(y)

    # *mean*
    print(y.mean(dim=(0, 2, 3)))
    # *std*
    print(y.std(dim=(0, 2, 3)))
