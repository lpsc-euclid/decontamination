########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import unittest

import numpy as np

from decontamination import jit

########################################################################################################################

@jit.jit(device = False)
def foo_xpu(a, b):

    return a + b

########################################################################################################################

# noinspection PyUnresolvedReferences
class JITTests(unittest.TestCase):

    ####################################################################################################################

    def test1(self):

        print('Running foo_xpu...')

        a = np.array([1, 2, 3, 4], dtype = np.float32)
        b = np.array([5, 6, 7, 8], dtype = np.float32)
        c = np.array([6, 8, 10, 12], dtype = np.float32)

        self.assertTrue(np.array_equal(foo_xpu(a, b), c))

    ####################################################################################################################

    def test2(self):

        if jit.CPU_OPTIMIZATION_AVAILABLE:

            print('Running foo_cpu...')

            a = np.array([1, 2, 3, 4], dtype = np.float32)
            b = np.array([5, 6, 7, 8], dtype = np.float32)
            c = np.array([6, 8, 10, 12], dtype = np.float32)

            self.assertTrue(np.array_equal(foo_cpu(a, b), c))

        else:

            print('Skip foo_cpu...')

    ####################################################################################################################

    def test3(self):

        if jit.GPU_OPTIMIZATION_AVAILABLE:

            print('Running foo_gpu...')

            a = np.array([1, 2, 3, 4], dtype = np.float32)
            b = np.array([5, 6, 7, 8], dtype = np.float32)
            c = np.array([6, 8, 10, 12], dtype = np.float32)

            self.assertTrue(np.array_equal(foo_gpu[(c.size + (32 - 1)) // 32, 32](a, b), c))

        else:

            print('Skip foo_gpu...')

########################################################################################################################

if __name__ == '__main__':

    unittest.main()

########################################################################################################################
