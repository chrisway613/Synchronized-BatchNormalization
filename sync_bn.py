# -*- coding: utf-8 -*-
# File   : sync_bn.py
# Author : CW
# Email  : chrisway613@gmail.com
# Date   : 21/01/2020
#
# This file is part of Synchronized-BatchNorm-PyTorch.

__all__ = (
    'SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d'
)

try:
    from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
except ImportError:
    ReduceAddCoalesced = Broadcast = None
from torch.nn.modules.batchnorm import _BatchNorm

try:
    from utils import *
    from sync import SyncMaster
    from parallel import DataParallelWithCallBack
except ImportError:
    from .utils import *
    from .sync import SyncMaster
    from .parallel import DataParallelWithCallBack

import collections

import torch
import torch.nn.functional as F


_MessageToCollect = collections.namedtuple('_ChildMessage', ('sum', 'ssum', 'sum_size'))
_MessageToBroadcast = collections.namedtuple('_MasterMessage', ('mean', 'inv_std'))


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, sync_timeout=15.):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'

        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._is_parallel = False
        self._parallel_id = None

        self._sync_master = SyncMaster(callback=self._coalesce_and_compute, sync_timeout=sync_timeout)
        self._slave_pipe = None

    @property
    def _is_master(self):
        assert self._parallel_id is not None, "parallel replicate method should be executed first!"
        return self._parallel_id == 0

    def forward(self, inputs):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                inputs, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps
            )

        inputs_shape = inputs.shape
        # Reshape to (N, C, -1), whereas N is batch size, C is number of features/classes.
        inputs = inputs.reshape(inputs_shape[0], self.num_features, -1)
        # Compute the sum and square-sum.
        sum_size = inputs.size(0) * inputs.size(2)
        input_sum = sum_ft(inputs)
        input_ssum = sum_ft(inputs ** 2)
        # Master will collect message as below from all copies.
        msg = _MessageToCollect(input_sum, input_ssum, sum_size)
        # Reduce & broadcast the statistics.
        if self._is_master:
            # print("run master\n")
            result = self._sync_master.run_master(msg)

            # When timeout occurred during synchronizing with slaves,
            # the result will be None,
            # then use PyTorch's implementation.
            if result is None:
                return F.batch_norm(
                    inputs, self.running_mean, self.running_var, self.weight, self.bias,
                    self.training, self.momentum, self.eps
                )
            else:
                mean, inv_std = result
        else:
            # print("run slave\n")
            result_from_master = self._slave_pipe.run_slave(msg)

            # When timeout occurred during synchronizing with master,
            # the result from master will be None,
            # then use PyTorch's implementation.
            if result_from_master is None:
                return F.batch_norm(
                    inputs, self.running_mean, self.running_var, self.weight, self.bias,
                    self.training, self.momentum, self.eps
                )
            else:
                mean, inv_std = result_from_master

        # Compute the output.
        if self.affine:
            outputs = (inputs - unsqueeze_ft(mean)) * unsqueeze_ft(inv_std * self.weight) + unsqueeze_ft(self.bias)
        else:
            outputs = (inputs - unsqueeze_ft(mean)) * unsqueeze_ft(inv_std)

        # Reshape to original input shape
        return outputs.reshape(inputs_shape)

    def _sync_replicas(self, ctx, copy_id):
        """
        Synchronize all copies from a module.
        :param ctx: a context data structure for communication;
        :param copy_id: id of a copied module (usually the device id).
        :return:
        """
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _coalesce_and_compute(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Ensure that master being the first one.
        intermediates = sorted(intermediates, key=lambda i: i[0])

        # Get sum & square sum of from every device.
        to_reduce = [i[1][:2] for i in intermediates]
        # Flatten
        to_reduce = [j for i in to_reduce for j in i]
        # Size of data from every device.
        sum_size = sum([i[1].sum_size for i in intermediates])
        # Device of every copies
        target_gpus = [i[1].sum.get_device() for i in intermediates]
        # print("target gpus: ", target_gpus)
        
        # Add all sum & square sum individually from every copies,
        # and put the result to the master device.
        # 2 means that has 2 types input data.
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)
        # Copied results for every device that to broadcasted.
        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)
        # print("broadcasted: ", broadcasted)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MessageToBroadcast(*broadcasted[i*2:i*2+2])))

        # print("outputs: ", outputs)
        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """
        Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device.
        """
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1!'

        def _compute():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        mean = sum_ / size
        sum_var = ssum - sum_ * mean
        unbias_var = sum_var / (size - 1)
        bias_var = sum_var / size

        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                _compute()
        else:
            _compute()

        return mean, bias_var.clamp(self.eps) ** -.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> inputs = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(inputs)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                'expected 2D or 3D input (got {}D input)'.format(input.dim())
            )


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> inputs = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> outputs = m(inputs)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(
                'expected 4D input (got {}D input)'.format(input.dim())
            )


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> inputs = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(inputs)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(
                'expected 5D input (got {}D input)'.format(input.dim())
            )
