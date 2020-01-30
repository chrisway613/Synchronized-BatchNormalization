# -*- coding: utf-8 -*-
# File   : utils.py
# Author : CW
# Email  : chrisway613@gmail.com
# Date   : 21/01/2020
#
# This file is part of Synchronized-BatchNorm-PyTorch.

__all__ = ('FutureResult', 'sum_ft', 'unsqueeze_ft')

import threading


class FutureResult:
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self, wait_timeout=30.):
        self._wait_timeout = wait_timeout

        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has not been fetched!'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait(timeout=self._wait_timeout)

            res = self._result
            self._result = None

            return res


def sum_ft(tensor):
    """sum over the first and last dimension"""
    return tensor.sum(dim=0).sum(dim=-1)


def unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)
