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

@decontamination.jit(gpu_kernel = True)
def foo_kernel_gpu(result, a, b):

    i = cu.grid(1)

    if i < result.shape[0]:

        # noinspection PyUnresolvedReferences
        result[i] = foo_gpu(a[i], b[i])

########################################################################################################################

def test_xpu():

    # noinspection PyUnresolvedReferences
    assert np.array_equal(foo_xpu(A, B), C)

########################################################################################################################

def test_cpu():

    # noinspection PyUnresolvedReferences
    assert np.array_equal(foo_cpu(A, B), C)

########################################################################################################################

def test_gpu():

    if decontamination.GPU_OPTIMIZATION_AVAILABLE:

        result = cu.device_array_like(C)

        foo_kernel_gpu[32, result.size](result, A, B)

        assert np.array_equal(result.copy_to_host(), C)

########################################################################################################################

if __name__ == '__main__':

    pytest.main([__file__])

########################################################################################################################
