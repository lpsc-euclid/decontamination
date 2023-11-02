########################################################################################################################

import typing
import functools
import threading

import numpy as np

########################################################################################################################

from numba import types

from numba.extending import lower_builtin
from numba.extending import type_callable

from numba.np.arrayobj import make_array
from numba.np.arrayobj import basic_indexing
from numba.np.arrayobj import normalize_indices

from numba.core.typing.arraydecl import get_array_index_type

########################################################################################################################

_global_lock = threading.Lock()

########################################################################################################################

def declare_atomic_array_op(iop: str, uop: str, fop: str):

    ####################################################################################################################

    def decorator(func):

        ################################################################################################################

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            with _global_lock:

                func(*args, **kwargs)

        ################################################################################################################

        @type_callable(wrapper)
        def func_type(context):

            ############################################################################################################

            def typer(array, index, value):

                out = get_array_index_type(array, index)

                if out is not None:

                    if context.can_convert(value, out.result):

                        return out.result

                return None

            ############################################################################################################

            return typer

        ################################################################################################################

        @lower_builtin(wrapper, types.Buffer, types.Any, types.Any)
        def func_impl(context, builder, signature, args):

            ############################################################################################################

            array_type, index_type, value_type = signature.args
            array     , index     , value      =           args

            ############################################################################################################

            index_types = (index_type, )

            indices = (index, )

            ############################################################################################################

            array = make_array(array_type)(context, builder, array)

            index_types, indices = normalize_indices(context, builder, index_types, indices)

            ptr, shapes, _ = basic_indexing(context, builder, array_type, array, index_types, indices, boundscheck = (context.enable_boundscheck, ))

            ############################################################################################################

            if shapes:

                raise NotImplementedError('Complex shapes are not supported')

            ############################################################################################################

            value = context.cast(builder, value, value_type, array_type.dtype)

            ############################################################################################################

            if isinstance(array_type.dtype, types.Float):
                op = fop
            elif isinstance(array_type.dtype, types.Integer) and array_type.dtype.signed:
                op = iop
            elif isinstance(array_type.dtype, types.Integer) and not array_type.dtype.signed:
                op = uop
            else:
                raise TypeError(f'Atomic operation not supported on {array_type}')

            ############################################################################################################

            return builder.atomic_rmw(op, ptr, context.get_value_as_data(builder, array_type.dtype, value), 'monotonic')

        ################################################################################################################

        return wrapper

    ####################################################################################################################

    return decorator

########################################################################################################################

@declare_atomic_array_op('add', 'add', 'fadd')
def add(array: np.ndarray, i: int, v: typing.Union[np.float64, np.float32, float, np.int64, np.int32, int]) -> typing.Union[np.float64, np.float32, float, np.int64, np.int32, int]:

    orig = array[i]
    array[i] += v
    return orig

########################################################################################################################

@declare_atomic_array_op('sub', 'sub', 'fsub')
def sub(array: np.ndarray, i: int, v: typing.Union[np.float64, np.float32, float, np.int64, np.int32, int]) -> typing.Union[np.float64, np.float32, float, np.int64, np.int32, int]:

    orig = array[i]
    array[i] -= v
    return orig

########################################################################################################################
