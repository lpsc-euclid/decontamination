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

A = np.array([1, 2,  3,  4], dtype = np.float32)
B = np.array([5, 6,  7,  8], dtype = np.float32)
C = np.array([6, 8, 10, 12], dtype = np.float32)

########################################################################################################################

@decontamination.jit(device = True)
def foo_xpu(a, b):

    return a + b

########################################################################################################################

@decontamination.jit(kernel = True)
def foo_kernel_gpu(result, a, b):

    tx = cu.threadIdx.x
    ty = cu.blockIdx.x
    bw = cu.blockDim.x
    pos = tx + ty * bw

    if pos < result.shape[0]:

        result[pos] = foo_gpu(a[pos], b[pos])

########################################################################################################################

# noinspection PyUnresolvedReferences
class JITTests(unittest.TestCase):

    ####################################################################################################################

    def test1(self):

        print('Running foo_xpu...')

        self.assertTrue(np.array_equal(foo_xpu(A, B), C))

    ####################################################################################################################

    def test2(self):

        if decontamination.CPU_OPTIMIZATION_AVAILABLE:

            print('Running foo_cpu...')

            self.assertTrue(np.array_equal(foo_cpu(A, B), C))

        else:

            print('Skip foo_cpu...')

    ####################################################################################################################

    def test3(self):

        if decontamination.GPU_OPTIMIZATION_AVAILABLE:

            print('Running foo_gpu...')

            r = np.zeros(C.size, dtype = np.float32)

            foo_kernel_gpu[(C.size + (32 - 1)) // 32, 32](r, A, B)

            self.assertTrue(np.array_equal(r, C))

        else:

            print('Skip foo_gpu...')

########################################################################################################################

if __name__ == '__main__':

    unittest.main()

########################################################################################################################
