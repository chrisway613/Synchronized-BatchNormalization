# -*- coding: utf-8 -*-
# File   : parallel.py
# Author : CW
# Email  : chrisway613@gmail.com
# Date   : 21/01/2020
#
# This file is part of Synchronized-BatchNorm-PyTorch.

__all__ = ('DataParallelWithCallBack',)

from torch.nn.parallel.data_parallel import DataParallel


class DataParallelContext:
    """
    Context data structure for data parallel.
    Multiple copies of a module on different devices share the same context,
    Thus with this context, different copies can share some information.
    """
    def __init__(self):
        self.sync_master = None


class DataParallelWithCallBack(DataParallel):
    """
        Data Parallel with a replication callback.

        An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
        original `replicate` function.
        The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

        Examples:
            > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
            > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
            # sync_bn.sync_replicas will be invoked.
        """
    @classmethod
    def _callback(cls, replicas):
        master_copy = replicas[0]
        replicas_ctx = [DataParallelContext() for _ in master_copy.modules()]

        for copy_id, module_replicated in enumerate(replicas):
            for idx, m in enumerate(module_replicated.modules()):
                if 'SynchronizedBatchNorm' in type(m).__name__ and hasattr(m, '_sync_replicas'):
                    m._sync_replicas(replicas_ctx[idx], copy_id)

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        """
        Initialization.
        :param module: module to be parallelized;
        :param device_ids: CUDA devices (default: all devices);
        :param output_device: device location of output (default: device_ids[0]);
        :param dim: dim of input data to be scattered & gathered.
        """
        super(DataParallelWithCallBack, self).__init__(
            module, device_ids, output_device, dim
        )

    def replicate(self, module, device_ids):
        """
        Replication with callback.
        :param module: (nn.Module) module to be parallelized;
        :param device_ids: (list of int or torch.device) CUDA devices (default: all devices);
        :return: module replicated on each device.
        """
        replicas = super(DataParallelWithCallBack, self).replicate(module, device_ids)
        self._callback(replicas)

        return replicas

    def forward(self, *inputs, **kwargs):
        """
        Note that this method will invoke the methods as below(in order):
        i). self.scatter;
        ii). self.replicate;
        iii). self.parallel_apply;
        iv). self.gather
        """
        return super(DataParallelWithCallBack, self).forward(*inputs, **kwargs)
