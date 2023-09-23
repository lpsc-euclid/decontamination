#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import pytest
import decontamination

import numpy as np

import numba.cuda as cu

########################################################################################################################

A = np.random.randn(100_000).astype(np.float32)
B = np.random.randn(100_000).astype(np.float32)
C = np.add(A, B)

########################################################################################################################

@decontamination.jit()
def foo_xpu(a, b):

    return a + b

########################################################################################################################

@decontamination.jit(kernel = True)
def foo_kernel_xpu(result, a, b):

    ####################################################################################################################
    # !--BEGIN-CPU--

    for i in range(result.shape[0]):

        # noinspection PyUnresolvedReferences
        result[i] = foo_cpu(a[i], b[i])

    # !--END-CPU--
    ####################################################################################################################
    # !--BEGIN-GPU--

    i = cu.grid(1)
    if i < result.shape[0]:

        # noinspection PyUnresolvedReferences
        result[i] = foo_gpu(a[i], b[i])

    # !--END-GPU--

########################################################################################################################

def test_xpu():

    assert np.array_equal(3, foo_xpu(1, 2))

########################################################################################################################

def test_cpu():

    result = np.zeros_like(C)

    # noinspection PyUnresolvedReferences
    foo_kernel_cpu[32, result.size](result, A, B)

    assert np.array_equal(result, C)

########################################################################################################################

def test_gpu():

    if decontamination.GPU_OPTIMIZATION_AVAILABLE:

        result = cu.device_array_like(C)

        # noinspection PyUnresolvedReferences
        foo_kernel_gpu[32, result.size](result, A, B)

        assert np.array_equal(result.copy_to_host(), C)

########################################################################################################################

if __name__ == '__main__':

    pytest.main([__file__])

########################################################################################################################
