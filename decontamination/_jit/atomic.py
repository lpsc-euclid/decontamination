########################################################################################################################

import functools
import threading

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

            def typer(ary, idx, val):

                out = get_array_index_type(ary, idx)

                if out is not None:

                    result = out.result

                    if context.can_convert(val, result):

                        return result

                return None

            ############################################################################################################

            return typer

        ################################################################################################################

        @lower_builtin(wrapper, types.Buffer, types.Any, types.Any)
        def func_impl(context, builder, signature, args):

            ############################################################################################################

            aryty, idxty, valty = signature.args

            ary, idx, val = args

            ############################################################################################################

            assert aryty.aligned

            ############################################################################################################

            index_types = (idxty, )

            indices = (idx, )

            ############################################################################################################

            ary = make_array(aryty)(context, builder, ary)

            index_types, indices = normalize_indices(context, builder, index_types, indices)

            ptr, shapes, _ = basic_indexing(context, builder, aryty, ary, index_types, indices, boundscheck = (context.enable_boundscheck, ))

            ############################################################################################################

            if shapes:

                raise NotImplementedError('Complex shapes are not supported')

            ############################################################################################################

            val = context.cast(builder, val, valty, aryty.dtype)

            ############################################################################################################

            if isinstance(aryty.dtype, types.Float):
                op = fop
            elif isinstance(aryty.dtype, types.Integer) and aryty.dtype.signed:
                op = iop
            elif isinstance(aryty.dtype, types.Integer) and not aryty.dtype.signed:
                op = uop
            else:
                raise TypeError(f'Atomic operation not supported on {aryty}')

            ############################################################################################################

            return builder.atomic_rmw(op, ptr, context.get_value_as_data(builder, aryty.dtype, val), 'monotonic')

        ################################################################################################################

        return wrapper

    ####################################################################################################################

    return decorator

########################################################################################################################

@declare_atomic_array_op('add', 'add', 'fadd')
def add(ary, i, v):

    orig = ary[i]
    ary[i] += v
    return orig

########################################################################################################################

@declare_atomic_array_op('sub', 'sub', 'fsub')
def sub(ary, i, v):

    orig = ary[i]
    ary[i] -= v
    return orig

########################################################################################################################
