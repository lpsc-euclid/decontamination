#!/usr/bin/env python3 -m pytest
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import pytest
import decontamination

import numpy as np

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
def foo_kernel(result, a, b):

    if decontamination.jit.is_gpu:

        ################################################################################################################
        # GPU CODE                                                                                                     #
        ################################################################################################################

        i = decontamination.jit.grid(1)

        if i < result.shape[0]:

            result[i] = foo_xpu(a[i], b[i])

        ################################################################################################################

    else:

        ################################################################################################################
        # CPU CODE                                                                                                     #
        ################################################################################################################

        for i in range(result.shape[0]):

            result[i] = foo_xpu(a[i], b[i])

########################################################################################################################

def test_xpu():

    with pytest.raises(RuntimeError):

        foo_xpu(1, 2)

########################################################################################################################

def test_cpu():

    result = decontamination.device_array_empty(C.shape, dtype = C.dtype)

    foo_kernel[False, 32, result.shape[0]](result, A, B)

    assert np.array_equal(result.copy_to_host(), C)

########################################################################################################################

def test_gpu():

    result = decontamination.device_array_empty(C.shape, dtype = C.dtype)

    foo_kernel[True, 32, result.shape[0]](result, A, B)

    assert np.array_equal(result.copy_to_host(), C)

########################################################################################################################
