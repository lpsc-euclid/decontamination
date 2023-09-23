# -*- coding: utf-8 -*-
########################################################################################################################

import os
import re
import typing
import inspect

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

# noinspection PyPep8Naming
class result_array(object):

    """
    Empty device ndarray to be used as result when calling a CPU/GPU kernel. Similar to `numpy.empty`.
    """

    ####################################################################################################################

    def __init__(self, shape: typing.Union[typing.Tuple[int], int], dtype: typing.Type[np.single] = np.float32):

        """
        Parameters
        ----------
        shape : typing.Union[typing.Tuple[int], int]
            Desired shape for the new array.
        dtype : typing.Type[np.single]
            Desired data-type for the new array.
        """

        self._shape = shape
        self._dtype = dtype

        self._instance = None

    ####################################################################################################################

    @property
    def shape(self):

        """
        Shape for the array.
        """

        return self._shape

    ####################################################################################################################

    @property
    def dtype(self):

        """
        Data-type for the array.
        """

        return self._dtype

    ####################################################################################################################

    def _instantiate(self, is_gpu: bool):

        ################################################################################################################

        if self._instance is not None:

            raise Exception('Array already instanced')

        ################################################################################################################

        if is_gpu:

            self._instance = cu.device_array(shape = self._shape, dtype = self._dtype)

        else:

            self._instance = np.empty(shape = self._shape, dtype = self._dtype)

        ################################################################################################################

        return self._instance

    ####################################################################################################################

    def copy_to_host(self) -> np.ndarray:

        """
        Create a new Numpy ndarray from the underlying device ndarray.
        """

        ################################################################################################################

        if self._instance is None:

            raise Exception('Array not instanced')

        ################################################################################################################

        if self._instance.__class__.__name__ == 'DeviceNDArray':

            return self._instance.copy_to_host()

        else:

            return self._instance

########################################################################################################################

class Kernel:

    ####################################################################################################################

    def __init__(self, cpu_func: typing.Callable, gpu_func: typing.Callable, parallel: bool):

        self.cpu_func = nb.njit(cpu_func, parallel = parallel) if CPU_OPTIMIZATION_AVAILABLE else cpu_func

        self.gpu_func = cu.jit(gpu_func, device = False) if GPU_OPTIMIZATION_AVAILABLE else gpu_func

    ####################################################################################################################

    def __getitem__(self, extra_params):

        ################################################################################################################

        if not isinstance(extra_params, tuple) or len(extra_params) != 3 or not isinstance(extra_params[0], bool):

            raise ValueError('Three parameters expected: run_on_gpu, threads_per_blocks and data_sizes')

        ################################################################################################################

        threads_per_blocks = extra_params[1] if isinstance(extra_params[1], tuple) else (extra_params[1], )

        data_sizes = extra_params[2] if isinstance(extra_params[2], tuple) else (extra_params[2], )

        num_blocks = tuple((s + t - 1) // t for s, t in zip(data_sizes, threads_per_blocks))

        ################################################################################################################

        def wrapper(*args, **kwargs):

            new_args = []

            if extra_params[0] and GPU_OPTIMIZATION_AVAILABLE:

                ########################################################################################################

                for arg in args:

                    if isinstance(arg, np.ndarray):
                        new_args.append(cu.to_device(arg))
                    elif isinstance(arg, result_array):
                        # noinspection PyProtectedMember
                        new_args.append(arg._instantiate(True))
                    else:
                        new_args.append(arg)

                ########################################################################################################

                return self.gpu_func[num_blocks, threads_per_blocks](*new_args, **kwargs)

                ########################################################################################################

            else:

                if extra_params[0]:

                    print('Will emulate GPU kernel...')

                ########################################################################################################

                for arg in args:

                    if isinstance(arg, np.ndarray):
                        new_args.append((((((((arg))))))))
                    elif isinstance(arg, result_array):
                        # noinspection PyProtectedMember
                        new_args.append(arg._instantiate(False))
                    else:
                        new_args.append(arg)

                ########################################################################################################

                return self.cpu_func(*new_args, **kwargs)

        ################################################################################################################

        return wrapper

########################################################################################################################

# noinspection PyPep8Naming,PyUnresolvedReferences
class jit(object):

    """
    Decorator to compile Python functions into native CPU/GPU ones.
    """

    ####################################################################################################################

    def __init__(self, kernel: bool = False, parallel: bool = False):

        """
        Parameters
        ---------
        kernel : bool
            Indicates whether this function is a CPU/GPU kernel (default: **False**).
        parallel : bool
            Enables automatic parallelization when running on CPU (default: **False**).
        """

        self.kernel = kernel
        self.parallel = parallel

    ####################################################################################################################

    _cnt = 0

    @classmethod
    def _get_unique_function_name(cls) -> str:

        name = f'__jit_f{cls._cnt}'

        cls._cnt = cls._cnt + 1

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

        if not self.kernel and not funct.__name__.endswith('_xpu'):

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

        if not self.kernel:

            jit._inject_cpu(funct, nb.njit(funct_cpu, parallel = self.parallel) if CPU_OPTIMIZATION_AVAILABLE else funct_cpu)

        ################################################################################################################
        # NUMBA ON CPU                                                                                                 #
        ################################################################################################################

        name_gpu = jit._get_unique_function_name()

        code_gpu = jit._patch_gpu_code(f'def {name_gpu} {code_raw}')

        ################################################################################################################

        exec(code_gpu, funct.__globals__)

        funct_gpu = eval(name_gpu, funct.__globals__)

        if not self.kernel:

            jit._inject_gpu(funct, cu.jit(funct_gpu, device = True) if GPU_OPTIMIZATION_AVAILABLE else funct_gpu)

        ################################################################################################################
        # KERNEL                                                                                                       #
        ################################################################################################################

        if self.kernel:

            funct = Kernel(funct_cpu, funct_gpu, parallel = self.parallel)

        ################################################################################################################

        return funct

########################################################################################################################
