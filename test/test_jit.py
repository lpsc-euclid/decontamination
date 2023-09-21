#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import unittest
import decontamination

import numpy as np

import numba.cuda as cu

########################################################################################################################

A = np.random.randn(100_000).astype(np.float32)
B = np.random.randn(100_000).astype(np.float32)
C = np.add(A, B)

########################################################################################################################

@decontamination.jit(device = True)
def foo_xpu(a, b):

    return a + b

########################################################################################################################

@decontamination.jit(gpu_kernel = True)
def foo_kernel_gpu(result, a, b):

    i = cu.threadIdx.x + cu.blockIdx.x * cu.blockDim.x

    if i < result.shape[0]:

        result[i] = foo_gpu(a[i], b[i])

########################################################################################################################

class JITTests(unittest.TestCase):

    ####################################################################################################################

    def test1(self):

        print('Running foo_xpu...')

        self.assertTrue(np.array_equal(foo_xpu(A, B), C))

    ####################################################################################################################

    def test2(self):

        print('Running foo_cpu...')

        self.assertTrue(np.array_equal(foo_cpu(A, B), C))

    ####################################################################################################################

    def test3(self):

        if decontamination.GPU_OPTIMIZATION_AVAILABLE:

            print('Running foo_gpu...')

            result = cu.device_array_like(C)

            foo_kernel_gpu[32, result.size](result, A, B)

            self.assertTrue(np.array_equal(result.copy_to_host(), C))

        else:

            print('Skipping foo_gpu...')

########################################################################################################################

if __name__ == '__main__':

    unittest.main()

########################################################################################################################
