# -*- coding: utf-8 -*-
########################################################################################################################

import os
import inspect
import functools

import numpy as np
import numba as nb

import numba.cuda as cu

########################################################################################################################

__pdoc__ = {}

########################################################################################################################

CPU_OPTIMIZATION_AVAILABLE = os.environ.get('FOO_USE_NUMBA', '1') != '0'
__pdoc__['CPU_OPTIMIZATION_AVAILABLE'] = 'Indicates whether the numba CPU optimization is available.'

########################################################################################################################

GPU_OPTIMIZATION_AVAILABLE = CPU_OPTIMIZATION_AVAILABLE and cu.is_available()
__pdoc__['GPU_OPTIMIZATION_AVAILABLE'] = 'Indicates whether the numba GPU optimization is available.'

########################################################################################################################

class DecoratedFunction:

    ####################################################################################################################

    def __init__(self, func):

        self.func = func

    ####################################################################################################################

    def __getitem__(self, extra_params):

        ####################################################################################################################

        if not isinstance(extra_params, tuple) or len(extra_params) != 2:

            raise ValueError('Two parameters expected: threads_per_blocks and data_sizes')

        ####################################################################################################################

        threads_per_blocks = extra_params[0] if isinstance(extra_params[0], tuple) else (extra_params[0], )

        data_sizes = extra_params[1] if isinstance(extra_params[1], tuple) else (extra_params[1], )

        num_blocks = tuple((s + t - 1) // t for s, t in zip(data_sizes, threads_per_blocks))

        ####################################################################################################################

        def wrapper(*args, **kwargs):

            args = [cu.to_device(arg) if isinstance(arg, np.ndarray) else arg for arg in args]

            return cu.jit(self.func, device = False)[num_blocks, threads_per_blocks](*args, **kwargs)

        ####################################################################################################################

        return wrapper

########################################################################################################################

# noinspection PyPep8Naming
class jit(object):

    """
    Decorator to compile Python functions into native CPU or GPU ones.
    """

    ####################################################################################################################

    def __init__(self, parallel: bool = False, cpu_kernel: bool = False, gpu_kernel: bool = False, device: bool = True):

        """
        Parameters
        ---------
        parallel : bool
            Enables automatic parallelization.
        kernel : bool
            Indicates whether this is a kernel function.
        device : bool
            Indicates whether this is a device function.
        """

        self.parallel = parallel
        self.cpu_kernel = cpu_kernel
        self.gpu_kernel = gpu_kernel
        self.device = device

    ####################################################################################################################

    _cnt = 0

    @classmethod
    def _get_unique_function_name(cls):

        name = f'__jit_f{cls._cnt}'

        cls.cnt = cls._cnt + 1

        return name

    ####################################################################################################################

    @staticmethod
    def _patch_cpu_code(code):

        return (
            code.replace('_xpu', '_cpu')
                .replace('xpu.local_empty', 'np.empty')
                .replace('xpu.shared_empty', 'np.empty')
                .replace('xpu.syncthreads', '#######')
        )

    ####################################################################################################################

    @staticmethod
    def _patch_gpu_code(code):

        return (
            code.replace('_xpu', '_gpu')
                .replace('xpu.local_empty', 'cu.local.array')
                .replace('xpu.shared_empty', 'cu.shared.array')
                .replace('xpu.syncthreads', 'cu.syncthreads')
        )

    ####################################################################################################################

    def __call__(self, funct):

        if self.cpu_kernel:

            return nb.njit(funct)

        elif self.gpu_kernel:

            return DecoratedFunction(funct)

        elif not funct.__name__.endswith('_xpu'):

            raise Exception(f'Function `{funct.__name__}` name must ends with `_xpu`')

        ################################################################################################################
        # SOURCE CODE                                                                                                  #
        ################################################################################################################

        code_raw = '\n'.join(inspect.getsource(funct).splitlines())

        code_raw = code_raw[code_raw.find("def"):]
        code_raw = code_raw[code_raw.find("("):]

        ################################################################################################################
        # NUMBA ON CPU                                                                                                 #
        ################################################################################################################

        if CPU_OPTIMIZATION_AVAILABLE:

            ############################################################################################################

            name_cpu = jit._get_unique_function_name()

            code_cpu = jit._patch_cpu_code(f'def {name_cpu} {code_raw}')

            exec(code_cpu, funct.__globals__)

            ############################################################################################################

            funct.__globals__[funct.__name__.replace('_xpu', '_cpu')] = nb.njit(eval(name_cpu, funct.__globals__), parallel = self.parallel)

        else:

            funct.__globals__[funct.__name__.replace('_xpu', '_cpu')] = funct

        ################################################################################################################
        # NUMBA ON GPU                                                                                                 #
        ################################################################################################################

        if GPU_OPTIMIZATION_AVAILABLE:

            ############################################################################################################

            name_gpu = jit._get_unique_function_name()

            code_gpu = jit._patch_gpu_code(f'def {name_gpu} {code_raw}')

            exec(code_gpu, funct.__globals__)

            ############################################################################################################

            funct.__globals__[funct.__name__.replace('_xpu', '_gpu')] = cu.jit(eval(name_gpu, funct.__globals__), device = self.device)

        else:

            funct.__globals__[funct.__name__.replace('_xpu', '_gpu')] = funct

        ################################################################################################################

        return funct

########################################################################################################################
