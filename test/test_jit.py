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

            r = cu.device_array_like(C)
            a = cu.const.array_like(A)
            b = cu.const.array_like(B)

            foo_kernel_gpu[(C.size + (32 - 1)) // 32, 32](r, a, b)

            self.assertTrue(np.array_equal(r.copy_to_host(), C))

        else:

            print('Skip foo_gpu...')

########################################################################################################################

if __name__ == '__main__':

    unittest.main()

########################################################################################################################
