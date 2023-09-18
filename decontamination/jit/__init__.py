########################################################################################################################

import os

import inspect

import numba as nb

import numba.cuda as cu

########################################################################################################################

CPU_OPTIMIZATION_AVAILABLE = os.environ.get('FOO_USE_NUMBA', '1') != '0'

########################################################################################################################

GPU_OPTIMIZATION_AVAILABLE = CPU_OPTIMIZATION_AVAILABLE and cu.is_available()

########################################################################################################################

# noinspection PyPep8Naming
class jit(object):

    ####################################################################################################################

    def __init__(self, parallel = False, kernel = False, device = True):

        self.parallel = parallel
        self.kernel = kernel
        self.device = device

    ####################################################################################################################

    cnt = 0

    @classmethod
    def get_unique_function_name(cls):

        name = f'__jit_f{cls.cnt}'

        cls.cnt = cls.cnt + 1

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

        if self.kernel:

            return cu.jit(funct, device = False)

        elif not funct.__name__.endswith('_xpu'):

            raise Exception(f'Function `{funct.__name__}` name must ends with `_xpu`')

        ################################################################################################################
        # SOURCE CODE                                                                                                  #
        ################################################################################################################

        code_raw = '\n'.join(inspect.getsource(funct).splitlines()[1:])

        ################################################################################################################
        # NUMBA ON CPU                                                                                                 #
        ################################################################################################################

        if CPU_OPTIMIZATION_AVAILABLE:

            ############################################################################################################

            name_cpu = jit.get_unique_function_name()

            code_cpu = jit._patch_cpu_code(f'def {name_cpu} {code_raw[code_raw.find("("):]}')

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

            name_gpu = jit.get_unique_function_name()

            code_gpu = jit._patch_gpu_code(f'def {name_gpu} {code_raw[code_raw.find("("):]}')

            exec(code_gpu, funct.__globals__)

            ############################################################################################################

            funct.__globals__[funct.__name__.replace('_xpu', '_gpu')] = cu.jit(eval(name_gpu, funct.__globals__), device = self.device)

        else:

            funct.__globals__[funct.__name__.replace('_xpu', '_gpu')] = funct

        ################################################################################################################

        return funct

########################################################################################################################
