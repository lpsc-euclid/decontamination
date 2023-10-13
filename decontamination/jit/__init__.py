# -*- coding: utf-8 -*-
########################################################################################################################

import os
import re
import sys
import typing
import inspect

import numpy as np
import numba as nb
import numba.cuda as cu

from . import atomic

########################################################################################################################

__pdoc__ = {}

########################################################################################################################

CPU_OPTIMIZATION_AVAILABLE = os.environ.get('USE_NUMBA_CPU', '1') != '0'
__pdoc__['CPU_OPTIMIZATION_AVAILABLE'] = 'Indicates whether the numba CPU optimization is available.'

########################################################################################################################

GPU_OPTIMIZATION_AVAILABLE = os.environ.get('USE_NUMBA_GPU', '1') != '0'
__pdoc__['GPU_OPTIMIZATION_AVAILABLE'] = 'Indicates whether the numba GPU optimization is available.'

########################################################################################################################

if not cu.is_available():

    GPU_OPTIMIZATION_AVAILABLE = False

########################################################################################################################

def nb_to_device(ndarray):

    # for future needs

    return ndarray

########################################################################################################################

def device_array_from(array: np.ndarray):

    """
    New device array (see :class:`DeviceArray`), initialized from a Numpy ndarray.
    """

    return DeviceArray(array.shape, array.dtype, content = array)

########################################################################################################################

def device_array_empty(shape: typing.Union[tuple, int], dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32):

    """
    New device array (see :class:`DeviceArray`), not initialized. Similar to `numpy.empty()`.

    Parameters
    ----------
    shape : typing.Union[tuple, int]
        Desired shape for the new array.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]
        Desired data-type for the new array.
    """

    return DeviceArray(shape, dtype, content = None)

########################################################################################################################

def device_array_zeros(shape: typing.Union[tuple, int], dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32):

    """
    New device array (see :class:`DeviceArray`), filled with **0**. Similar to `numpy.zeros()`.

    Parameters
    ----------
    shape : typing.Union[tuple, int]
        Desired shape for the new array.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]
        Desired data-type for the new array.
    """

    return DeviceArray(shape, dtype, content = 0)

########################################################################################################################

def device_array_full(shape: typing.Union[tuple, int], value: typing.Union[int, float], dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32):

    """
    New device array (see :class:`DeviceArray`), filled with **value**. Similar to `numpy.full()`.

    Parameters
    ----------
    shape : typing.Union[tuple, int]
        Desired shape for the new array.
    value : typing.Union[int, float]
        Desired value for the new array.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]
        Desired data-type for the new array.
    """

    return DeviceArray(shape, dtype, content = value)

########################################################################################################################

class DeviceArray(object):

    """
    Device array to be used when calling a CPU/GPU kernel.

    Prefer using primitives :class:`device_array_from`, :class:`device_array_empty`, :class:`device_array_zeros`, :class:`device_array_full` to instantiate a device array.

    Parameters
    ----------
    shape : typing.Union[tuple, int]
        Desired shape for the new array.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]
        Desired data-type for the new array.
    content : typing.Optional[typing.Union[int, float, np.ndarray]]
        Optional content, integer, floating ot Numpy ndarray.
    """

    ####################################################################################################################

    def __init__(self, shape: typing.Union[tuple, int], dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, content: typing.Optional[typing.Union[int, float, np.ndarray]] = None):

        self._shape = shape
        self._dtype = dtype

        self._content = content

        self._instance = None

    ####################################################################################################################

    @property
    def shape(self):

        """
        Shape of the array.
        """

        return self._shape

    ####################################################################################################################

    @property
    def dtype(self):

        """
        Data-type of the array.
        """

        return self._dtype

    ####################################################################################################################

    def _instantiate(self, is_gpu: bool):

        ################################################################################################################

        if self._instance is not None:

            return self._instance

        ################################################################################################################

        if is_gpu:

            ############################################################################################################
            # GPU INSTANTIATION                                                                                        #
            ############################################################################################################

            if isinstance(self._content, np.ndarray):

                self._instance = cu.to_device(self._content)

            elif self._content is not None:

                if float(self._content) == 0.0:
                    self._instance = cu.to_device(np.zeros(self._shape, dtype = self._dtype))
                else:
                    self._instance = cu.to_device(np.full(self._shape, self._content, dtype = self._dtype))

            else:

                self._instance = cu.device_array(self._shape, dtype = self._dtype)

            ############################################################################################################

        else:

            ############################################################################################################
            # CPU INSTANTIATION                                                                                        #
            ############################################################################################################

            if isinstance(self._content, np.ndarray):

                self._instance = nb_to_device(self._content)

            elif self._content is not None:

                if float(self._content) == 0.0:
                    self._instance = np.zeros(self._shape, dtype = self._dtype)
                else:
                    self._instance = np.full(self._shape, self._content, dtype = self._dtype)

            else:

                self._instance = np.empty(self._shape, dtype = self._dtype)

        ################################################################################################################

        return self._instance

    ####################################################################################################################

    def copy_to_host(self) -> np.ndarray:

        """
        Create a new Numpy ndarray from the underlying device ndarray.
        """

        ################################################################################################################

        if self._instance is None:

            raise Exception('Device array not instanced')

        ################################################################################################################

        if cu.is_cuda_array(self._instance):

            return self._instance.copy_to_host()

        else:

            return self._instance

########################################################################################################################

# noinspection PyUnusedLocal
def dont_call(*args, **kwargs):

    raise RuntimeError('D\'ont call me')

########################################################################################################################

class Kernel:

    ####################################################################################################################

    def __init__(self, cpu_func: typing.Callable, gpu_func: typing.Callable, fastmath: bool, parallel: bool):

        self.cpu_func = nb.njit(cpu_func, fastmath = fastmath, parallel = parallel) if CPU_OPTIMIZATION_AVAILABLE else cpu_func

        self.gpu_func = cu.jit(gpu_func, device = False) if GPU_OPTIMIZATION_AVAILABLE else dont_call

    ####################################################################################################################

    def __getitem__(self, kernel_params):

        ################################################################################################################

        if not isinstance(kernel_params, tuple) or len(kernel_params) != 3 or not isinstance(kernel_params[0], bool):

            raise ValueError('Three kernel parameters are expected: run_on_gpu, threads_per_blocks and data_sizes')

        ################################################################################################################

        threads_per_blocks = kernel_params[1] if isinstance(kernel_params[1], tuple) else (kernel_params[1], )

        data_sizes = kernel_params[2] if isinstance(kernel_params[2], tuple) else (kernel_params[2], )

        num_blocks = tuple((s + t - 1) // t for s, t in zip(data_sizes, threads_per_blocks))

        ################################################################################################################

        def wrapper(*args, **kwargs):

            new_args = []

            if kernel_params[0] and GPU_OPTIMIZATION_AVAILABLE:

                ########################################################################################################
                # RUN GPU KERNEL                                                                                       #
                ########################################################################################################

                for arg in args:

                    if isinstance(arg, np.ndarray):
                        new_args.append(cu.to_device(arg))
                    elif isinstance(arg, DeviceArray):
                        # noinspection PyProtectedMember
                        new_args.append(arg._instantiate(True))
                    else:
                        new_args.append(arg)

                ########################################################################################################

                return self.gpu_func[num_blocks, threads_per_blocks](*new_args, **kwargs)

                ########################################################################################################

            else:

                if kernel_params[0]:

                    print('Will emulate a GPU kernel...', file = sys.stderr, flush = True)

                ########################################################################################################
                # RUN CPU KERNEL                                                                                       #
                ########################################################################################################

                for arg in args:

                    if isinstance(arg, np.ndarray):
                        new_args.append(nb_to_device(arg))
                    elif isinstance(arg, DeviceArray):
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
    Decorator to recompile Python functions into native CPU/GPU ones.

    Parameters
    ----------
    kernel : bool
        Indicates whether this function is a CPU/GPU kernel (default: **False**).
    inline : bool
        Indicates whether this function must be inlined  (default: **False**).
    fastmath : bool
        Enables fast-math optimizations when running on CPU (default: **False**).
    parallel : bool
        Enables automatic parallelization when running on CPU (default: **False**).

    Example
    -------
        @jit()
        def foo_xpu(a, b):

            return a + b

        @jit(kernel = True)
        def foo_kernel(result, a, b):

            ########################################################################
            # !--BEGIN-CPU--

            for i in range(result.shape[0]):

                result[i] = foo_xpu(a[i], b[i])

            # !--END-CPU--
            ########################################################################
            # !--BEGIN-GPU--

            i = jit.grid(1)
            if i < result.shape[0]:

                result[i] = foo_xpu(a[i], b[i])

            # !--END-GPU--
            ########################################################################

        use_gpu = True
        threads_per_block = 32

        A = np.random.randn(100_000).astype(np.float32)
        B = np.random.randn(100_000).astype(np.float32)

        result = device_array_empty(100_000, dtype = np.float32)

        foo_kernel[use_gpu, threads_per_block, result.shape[0]](result, A, B)

        print(result.copy_to_host())
    """

    ####################################################################################################################

    @staticmethod
    def grid(ndim: int) -> typing.Union[tuple, int]:

        """
        Return the absolute position of the current thread in the entire grid of blocks. Use on GPU only.

        Parameters
        ----------
        ndim : int
            Number of dimensions.
        """

        pass

    ####################################################################################################################

    @staticmethod
    def local_empty(shape: typing.Union[tuple, int], dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32) -> np.ndarray:

        """
        Allocate an empty device ndarray in the local memory. Similar to `numpy.empty()` on CPU.

        Parameters
        ----------
        shape : typing.Union[tuple, int]
            Desired shape for the new array.
        dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]
            Desired data-type for the new array.
        """

        pass

    ####################################################################################################################

    @staticmethod
    def shared_empty(shape: typing.Union[tuple, int], dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32) -> np.ndarray:

        """
        Allocate an empty device ndarray in the shared memory. Similar to `numpy.empty()` on CPU.

        Parameters
        ----------
        shape : typing.Union[tuple, int]
            Desired shape for the new array.
        dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]
            Desired data-type for the new array.
        """

        pass

    ####################################################################################################################

    @staticmethod
    def syncthreads() -> None:

        """
        Synchronize all threads in the same thread block. Ignored on CPU.
        """

        pass

    ####################################################################################################################

    @staticmethod
    def atomic_add(array: np.ndarray, idx: int, val: typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]) -> typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]:

        """
        Performs atomic `array[idx] += val` and returns the old value. Supported on int32/64 and float32/64 operands only.

        Parameters
        ----------
        array : np.ndarray
            Array to be modified.
        idx : int
            Index in the array.
        val : typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]
            New value.
        """

        pass

    ####################################################################################################################

    @staticmethod
    def atomic_sub(array: np.ndarray, idx: int, val: typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]) -> typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]:

        """
        Performs atomic `array[idx] -= val` and returns the old value. Supported on int32/64 and float32/64 operands only.

        Parameters
        ----------
        array : np.ndarray
            Array to be modified.
        idx : int
            Index in the array.
        val : typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]
            New value.
        """

        pass

    ####################################################################################################################

    _CALL_RE = re.compile('(\\w+)_xpu\\s*\\(')

    _JIT_X_RE = re.compile('(?:\\w*\\.)?jit\\.')

    _METHOD_RE = re.compile('def[^(]+(\\(.*)', flags = re.DOTALL)

    _CPU_CODE_RE = re.compile(re.escape('!--BEGIN-CPU--') + '.*?' + re.escape('!--END-CPU--'), flags = re.DOTALL)

    _GPU_CODE_RE = re.compile(re.escape('!--BEGIN-GPU--') + '.*?' + re.escape('!--END-GPU--'), flags = re.DOTALL)

    ####################################################################################################################

    def __init__(self, kernel: bool = False, inline: bool = False, fastmath: bool = False, parallel: bool = False):

        self._kernel = kernel
        self._inline = inline
        self._fastmath = fastmath
        self._parallel = parallel

    ####################################################################################################################

    _cnt = 0

    @classmethod
    def _get_unique_function_name(cls) -> str:

        name = f'__jit_f{cls._cnt}'

        cls._cnt = cls._cnt + 1

        return name

    ####################################################################################################################

    @classmethod
    def _patch_cpu_code(cls, code: str) -> str:

        code_cpu = cls._CALL_RE.sub(lambda m: f'jit_module.{m.group(1)}_cpu(', cls._JIT_X_RE.sub('jit.', cls._GPU_CODE_RE.sub('', code)))

        return (
            code_cpu
            .replace('jit.grid', '# jit.grid')
            .replace('jit.local_empty', 'np.empty')
            .replace('jit.shared_empty', 'np.empty')
            .replace('jit.syncthreads', '# jit.syncthreads')
            .replace('jit.atomic_add', 'jit_module.atomic.add')
            .replace('jit.atomic_sub', 'jit_module.atomic.sub')
        )

    ####################################################################################################################

    @classmethod
    def _patch_gpu_code(cls, code: str) -> str:

        code_gpu = cls._CALL_RE.sub(lambda m: f'jit_module.{m.group(1)}_gpu(', cls._JIT_X_RE.sub('jit.', cls._CPU_CODE_RE.sub('', code)))

        return (
            code_gpu
            .replace('jit.grid', 'cuda_module.grid')
            .replace('jit.local_empty', 'cuda_module.local.array')
            .replace('jit.shared_empty', 'cuda_module.shared.array')
            .replace('jit.syncthreads', 'cuda_module.syncthreads')
            .replace('jit.atomic_add', 'cuda_module.atomic.add')
            .replace('jit.atomic_sub', 'cuda_module.atomic.sub')
        )

    ####################################################################################################################

    @classmethod
    def _inject_cpu_funct(cls, orig_funct: typing.Callable, new_funct: typing.Callable) -> None:

        name = orig_funct.__name__.replace('_xpu', '_cpu')

        if name in globals():

            raise Exception(f'Function {name} already declared')

        globals()[name] = new_funct

    ####################################################################################################################

    @classmethod
    def _inject_gpu_funct(cls, orig_funct: typing.Callable, new_funct: typing.Callable) -> None:

        name = orig_funct.__name__.replace('_xpu', '_gpu')

        if name in globals():

            raise Exception(f'Function {name} already declared')

        globals()[name] = new_funct

    ####################################################################################################################

    def __call__(self, funct: typing.Callable):

        if not self._kernel and not funct.__name__.endswith('_xpu'):

            raise Exception(f'Function `{funct.__name__}` name must ends with `_xpu`')

        ################################################################################################################
        # FRAME                                                                                                        #
        ################################################################################################################

        me = sys.modules[__name__]

        funct.__globals__['jit_module'] = me

        funct.__globals__['cuda_module'] = cu

        ################################################################################################################
        # SOURCE CODE                                                                                                  #
        ################################################################################################################

        code_raw = jit._METHOD_RE.search(inspect.getsource(funct)).group(1)

        ################################################################################################################
        # NUMBA ON GPU                                                                                                 #
        ################################################################################################################

        name_cpu = jit._get_unique_function_name()

        code_cpu = jit._patch_cpu_code(f'def {name_cpu} {code_raw}')

        ################################################################################################################

        exec(code_cpu, funct.__globals__)

        funct_cpu = eval(name_cpu, funct.__globals__)

        if not self._kernel:

            jit._inject_cpu_funct(funct, nb.njit(funct_cpu, inline = 'always' if self._inline else 'never', fastmath = self._fastmath, parallel = self._parallel) if CPU_OPTIMIZATION_AVAILABLE else funct_cpu)

        ################################################################################################################
        # NUMBA ON CPU                                                                                                 #
        ################################################################################################################

        name_gpu = jit._get_unique_function_name()

        code_gpu = jit._patch_gpu_code(f'def {name_gpu} {code_raw}')

        ################################################################################################################

        exec(code_gpu, funct.__globals__)

        funct_gpu = eval(name_gpu, funct.__globals__)

        if not self._kernel:

            jit._inject_gpu_funct(funct, cu.jit(funct_gpu, inline = self._inline, device = True) if GPU_OPTIMIZATION_AVAILABLE else dont_call)

        ################################################################################################################
        # KERNEL                                                                                                       #
        ################################################################################################################

        funct = Kernel(funct_cpu, funct_gpu, fastmath = self._fastmath, parallel = self._parallel) if self._kernel else dont_call

        ################################################################################################################

        return funct

########################################################################################################################
