# -*- coding: utf-8 -*-
# File   : func.py
# Author : CW
# Email  : chrisway613@gmail.com
# Date   : 21/01/2020
#
# This file is part of Synchronized-BatchNorm-PyTorch.

__all__ = ('patch_replication_callback', 'convert_model')

# +
try:
    from sync_bn import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
    from parallel import DataParallelWithCallBack
except ImportError:
    from .sync_bn import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
    from .parallel import DataParallelWithCallBack

from torch.nn import DataParallel, BatchNorm1d, BatchNorm2d, BatchNorm3d
# -

import torch
import functools


def convert_model(module):
    """
    Convert input module and its child recursively.
    :param module: the input module needs to be convert to SyncBN model;
    :return:
    Examples:
        >>> import torch.nn as nn
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = nn.DataParallel(m)
        >>> # after convert, m is using SyncBN
        >>> m = convert_model(m)
    """

    def _convert(mod_old):
        if 'BatchNorm' not in type(mod_old).__name__:
            return mod_old

        mod_new = mod_old
        for pth_module, sync_module in zip(
                [BatchNorm1d,
                 BatchNorm2d,
                 BatchNorm3d],
                [SynchronizedBatchNorm1d,
                 SynchronizedBatchNorm2d,
                 SynchronizedBatchNorm3d]
        ):
            if isinstance(mod_old, pth_module):
                mod_new = sync_module(mod_old.num_features, mod_old.eps, mod_old.momentum, mod_old.affine)
                mod_new.running_mean = mod_old.running_mean
                mod_new.running_var = mod_old.running_var

                if mod_old.affine:
                    mod_new.weight.data = mod_old.weight.data.clone().detach()
                    mod_new.bias.data = mod_old.bias.data.clone().detach()

        return mod_new

    if isinstance(module, torch.nn.DataParallel):
        # Top model inside DataParallel.
        mod = module.module
        mod = convert_model(mod)
        mod = DataParallelWithCallBack(mod, device_ids=module.device_ids)

        return mod

    mod_cvt = _convert(module)
    for name, child in module.named_children():
        mod_cvt.add_module(name, _convert(child))

    return mod_cvt


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)
    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        replicas = old_replicate(module, device_ids)
        # execute_replication_callbacks(modules)
        DataParallelWithCallBack._callback(replicas)

        return replicas

    data_parallel.replicate = new_replicate
