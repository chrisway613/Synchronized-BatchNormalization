# -*- coding: utf-8 -*-
# File   : sync.py
# Author : CW
# Email  : chrisway613@gmail.com
# Date   : 21/01/2020
# 
# This file is part of Synchronized-BatchNorm-PyTorch.

__all__ = ('FutureResult', 'SlavePipe', 'SyncMaster')

try:
    from utils import *
except ImportError:
    from .utils import *

import time
import queue
import collections


_Registry = collections.namedtuple('_Registry', ('result',))
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ('identifier', 'queue', 'result'))


class SlavePipe(_SlavePipeBase):
    """Pipe for master <=> slave communication."""
    def run_slave(self, msg):
        # Put msg to the queue which shared with master & all other slave copies.
        self.queue.put((self.identifier, msg))
        # Get result from master
        ret = self.result.get()
        # Notify master that result is already got.
        self.queue.put(True)

        return ret


class SyncMaster:
    """An abstract `SyncMaster` object.
    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """
    def __init__(self, callback=None, sync_timeout=15.):
        """
        Args:
            callback: a callback method to be invoked after having collected messages from slave devices.
        """
        self._callback = callback
        self._sync_timeout = sync_timeout

        self._activated = False
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()

    @property
    def num_slaves(self):
        return len(self._registry)

    def register_slave(self, identifier):
        """
        Register an slave device.
        The 'future' data structure stores slave's results;
        The '_registry' attribute records the mapping relation between slave's copy id & results;
        Master & its all copies share the same queue.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.
        """
        if self._activated:
            # assert self._queue.empty(), 'Queue is not cleaned before next initialization!'
            self._queue.queue.clear()
            self._activated = False
            self._registry.clear()

        future = FutureResult(wait_timeout=2*self._sync_timeout)
        self._registry[identifier] = _Registry(future)

        return SlavePipe(identifier, self._queue, future)

    def run_master(self, msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Note that if timeout occurred, this method will not be invoked.

        Args:
            msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, msg)]
        prev_time = time.time()
        # Until gather all slaves' msg or timeout occurred.
        while self._queue.qsize() != self.num_slaves:
            cur_time = time.time()
            time_used = cur_time - prev_time

            if time_used > self._sync_timeout:
                return None

        intermediates.extend([self._queue.get() for _ in range(self.num_slaves)])
        # print("intermediates: ", intermediates)
        results = self._callback(intermediates)
        # print(results)
        assert results[0][0] == 0, 'The first result should belongs to the master!'

        # results[0] belongs to master
        for i, res in results[1:]:
            # Return result to slave.
            self._registry[i].result.put(res)

        # Checkout whether slave has already got the result.
        for i in range(self.num_slaves):
            assert self._queue.get() is True

        # Return the result to master which belongs to itself.
        return results[0][1]
