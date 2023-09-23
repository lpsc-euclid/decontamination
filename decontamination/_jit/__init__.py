# -*- coding: utf-8 -*-
########################################################################################################################

import os
import re
import inspect
import typing

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

class NBKernel:

    ####################################################################################################################

    def __init__(self, func: typing.Callable, parallel: bool):

        self.func = nb.njit(func, parallel = parallel)

    ####################################################################################################################

    def __getitem__(self, extra_params):

        ################################################################################################################

        def wrapper(*args, **kwargs):

            return self.func(*args, **kwargs)

        ################################################################################################################

        return wrapper

########################################################################################################################

class CUKernel:

    ####################################################################################################################

    def __init__(self, func: typing.Callable, device: bool):

        self.func = cu.jit(func, device = device) if GPU_OPTIMIZATION_AVAILABLE else func

    ####################################################################################################################

    def __getitem__(self, extra_params):

        ################################################################################################################

        if not isinstance(extra_params, tuple) or len(extra_params) != 2:

            raise ValueError('Two parameters expected: threads_per_blocks and data_sizes')

        ################################################################################################################

        threads_per_blocks = extra_params[0] if isinstance(extra_params[0], tuple) else (extra_params[0], )

        data_sizes = extra_params[1] if isinstance(extra_params[1], tuple) else (extra_params[1], )

        num_blocks = tuple((s + t - 1) // t for s, t in zip(data_sizes, threads_per_blocks))

        ################################################################################################################

        def wrapper(*args, **kwargs):

            if GPU_OPTIMIZATION_AVAILABLE:

                args = [cu.to_device(arg) if isinstance(arg, np.ndarray) else arg for arg in args]

                return self.func[num_blocks, threads_per_blocks](*args, **kwargs)

            else:

                return self.func(*args, **kwargs)

        ################################################################################################################

        return wrapper

########################################################################################################################

# noinspection PyPep8Naming
class jit(object):

    """
    Decorator to compile Python functions into native CPU or GPU ones.
    """

    ####################################################################################################################

    def __init__(self, kernel: bool = False, parallel: bool = False):

        """
        Parameters
        ---------
        kernel : bool
            Indicates whether this is a kernel function.
        parallel : bool
            Enables automatic parallelization.
        """

        self.kernel = kernel
        self.parallel = parallel

    ####################################################################################################################

    _cnt = 0

    @classmethod
    def _get_unique_function_name(cls) -> str:

        name = f'__jit_f{cls._cnt}'

        cls.cnt = cls._cnt + 1

        return name

    ####################################################################################################################

    @staticmethod
    def _process_directives(code: str, tag_s: str, tag_e: str) -> str:

        pattern = re.compile(re.escape(tag_s) + '.*?' + re.escape(tag_e), re.DOTALL)

        return re.sub(pattern, '', code)

    ####################################################################################################################

    @staticmethod
    def _patch_cpu_code(code: str) -> str:

        return jit._process_directives(
            code.replace('_xpu', '_cpu')
                .replace('xpu.local_empty', 'np.empty')
                .replace('xpu.shared_empty', 'np.empty')
                .replace('xpu.syncthreads', '#######'),
            '!--BEGIN-GPU--',
            '!--END-GPU--'
        )

    ####################################################################################################################

    @staticmethod
    def _patch_gpu_code(code: str) -> str:

        if GPU_OPTIMIZATION_AVAILABLE:

            return jit._process_directives(
                code.replace('_xpu', '_gpu')
                    .replace('np.prange', 'range')
                    .replace('xpu.local_empty', 'cu.local.array')
                    .replace('xpu.shared_empty', 'cu.shared.array')
                    .replace('xpu.syncthreads', 'cu.syncthreads'),
                '!--BEGIN-CPU--',
                '!--END-CPU--'
            )

        else:

            return jit._process_directives(
                code.replace('_xpu', '_gpu')
                    .replace('np.prange', 'range')
                    .replace('xpu.local_empty', 'np.empty')
                    .replace('xpu.shared_empty', 'np.empty')
                    .replace('xpu.syncthreads', '#######'),
                '!--BEGIN-CPU--',
                '!--END-CPU--'
            )

    ####################################################################################################################

    @staticmethod
    def _inject_cpu(orig_funct: typing.Callable, new_funct: typing.Callable) -> None:

        orig_funct.__globals__[orig_funct.__name__.replace('_xpu', '_cpu')] = new_funct

    ####################################################################################################################

    @staticmethod
    def _inject_gpu(orig_funct: typing.Callable, new_funct: typing.Callable) -> None:

        orig_funct.__globals__[orig_funct.__name__.replace('_xpu', '_gpu')] = new_funct

    ####################################################################################################################

    def __call__(self, funct: typing.Callable):

        if not funct.__name__.endswith('_xpu'):

            raise Exception(f'Function `{funct.__name__}` name must ends with `_xpu`')

        ################################################################################################################
        # SOURCE CODE                                                                                                  #
        ################################################################################################################

        code_raw = '\n'.join(inspect.getsource(funct).splitlines())

        code_raw = code_raw[code_raw.find("def"):]

        code_raw = code_raw[code_raw.find("("):]

        ################################################################################################################
        # NUMBA ON GPU                                                                                                 #
        ################################################################################################################

        name_cpu = jit._get_unique_function_name()

        code_cpu = jit._patch_cpu_code(f'def {name_cpu} {code_raw}')

        ################################################################################################################

        exec(code_cpu, funct.__globals__)

        funct_cpu = eval(name_cpu, funct.__globals__)

        if self.kernel:
            jit._inject_cpu(funct, NBKernel(funct_cpu, parallel = self.parallel))
        else:
            jit._inject_cpu(funct, nb.njit(funct_cpu, parallel = self.parallel) if CPU_OPTIMIZATION_AVAILABLE else funct_cpu)

        ################################################################################################################
        # NUMBA ON CPU                                                                                                 #
        ################################################################################################################

        name_gpu = jit._get_unique_function_name()

        code_gpu = jit._patch_gpu_code(f'def {name_gpu} {code_raw}')

        ################################################################################################################

        exec(code_gpu, funct.__globals__)

        funct_gpu = eval(name_gpu, funct.__globals__)

        if self.kernel:
            jit._inject_gpu(funct, CUKernel(funct_gpu, device = False))
        else:
            jit._inject_gpu(funct, cu.jit(funct_gpu, device = True) if GPU_OPTIMIZATION_AVAILABLE else funct_gpu)

        ################################################################################################################

        return funct

########################################################################################################################
