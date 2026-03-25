# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import typing

import numpy as np

########################################################################################################################
# APPLY                                                                                                                #
########################################################################################################################

def _array_apply(f: typing.Callable, *arrays: np.ndarray) -> np.ndarray:

    if len({array.shape[0] for array in arrays}) > 1:

        raise ValueError('dataset chunks must be aligned')

    return f(*arrays)

########################################################################################################################

def _generator_factory_apply(f: typing.Callable, *datasets: typing.Callable) -> typing.Callable:

    # noinspection PyArgumentList
    def builder():

        for chunks in zip(*(dataset() for dataset in datasets)):

            yield _array_apply(f, *chunks)

    return builder

########################################################################################################################

def ds_apply(f: typing.Callable, *datasets: typing.Union[np.ndarray, typing.Callable]) -> typing.Union[np.ndarray, typing.Callable]:

    if not datasets:

        raise ValueError('at least one dataset is required')

    if all(isinstance(dataset, np.ndarray) for dataset in datasets):

        return _array_apply(f, *datasets)

    elif all(callable(dataset) for dataset in datasets):

        return _generator_factory_apply(f, *datasets)

    else:

        raise ValueError('datasets must all be arrays or all be callable')

########################################################################################################################
# ADD                                                                                                                  #
########################################################################################################################

def _array_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    if a.shape[0] != b.shape[0]:

        raise ValueError('`a` and `b` chunks must be aligned')

    return np.add(a, b)

########################################################################################################################

def _generator_factory_add(a: typing.Callable, b: typing.Callable) -> typing.Callable:

    def builder():

        for a_chunk, b_chunk in zip(a(), b()):

            yield _array_add(a_chunk, b_chunk)

    return builder

########################################################################################################################

def ds_add(a: typing.Union[np.ndarray, typing.Callable], b: typing.Union[np.ndarray, typing.Callable]) -> typing.Union[np.ndarray, typing.Callable]:

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):

        return _array_add(a, b)

    elif callable(a) and callable(b):

        return _generator_factory_add(a, b)

    else:

        raise ValueError("`a` and `b` must both be arrays or both be callable")

########################################################################################################################
# SUBTRACT                                                                                                             #
########################################################################################################################

def _array_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    if a.shape[0] != b.shape[0]:

        raise ValueError('`a` and `b` chunks must be aligned')

    return np.subtract(a, b)

########################################################################################################################

def _generator_factory_subtract(a: typing.Callable, b: typing.Callable) -> typing.Callable:

    def builder():

        for a_chunk, b_chunk in zip(a(), b()):

            yield _array_subtract(a_chunk, b_chunk)

    return builder

########################################################################################################################

def ds_subtract(a: typing.Union[np.ndarray, typing.Callable], b: typing.Union[np.ndarray, typing.Callable]) -> typing.Union[np.ndarray, typing.Callable]:

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):

        return _array_subtract(a, b)

    elif callable(a) and callable(b):

        return _generator_factory_subtract(a, b)

    else:

        raise ValueError("`a` and `b` must both be arrays or both be callable")

########################################################################################################################
# MULTIPLY                                                                                                             #
########################################################################################################################

def _array_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    if a.shape[0] != b.shape[0]:

        raise ValueError('`a` and `b` chunks must be aligned')

    return np.multiply(a, b)

########################################################################################################################

def _generator_factory_multiply(a: typing.Callable, b: typing.Callable) -> typing.Callable:

    def builder():

        for a_chunk, b_chunk in zip(a(), b()):

            yield _array_multiply(a_chunk, b_chunk)

    return builder

########################################################################################################################

def ds_multiply(a: typing.Union[np.ndarray, typing.Callable], b: typing.Union[np.ndarray, typing.Callable]) -> typing.Union[np.ndarray, typing.Callable]:

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):

        return _array_multiply(a, b)

    elif callable(a) and callable(b):

        return _generator_factory_multiply(a, b)

    else:

        raise ValueError("`a` and `b` must both be arrays or both be callable")

########################################################################################################################
# DIVIDE                                                                                                               #
########################################################################################################################

def _array_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    if a.shape[0] != b.shape[0]:

        raise ValueError('`a` and `b` chunks must be aligned')

    shape = np.broadcast_shapes(a.shape, b.shape)

    dtype = np.result_type(a, b)

    return np.divide(a, b, out = np.zeros(shape, dtype = dtype if np.issubdtype(dtype, np.floating) else np.float32), where = b != 0)

########################################################################################################################

def _generator_factory_divide(a: typing.Callable, b: typing.Callable) -> typing.Callable:

    def builder():

        for a_chunk, b_chunk in zip(a(), b()):

            yield _array_divide(a_chunk, b_chunk)

    return builder

########################################################################################################################

def ds_divide(a: typing.Union[np.ndarray, typing.Callable], b: typing.Union[np.ndarray, typing.Callable]) -> typing.Union[np.ndarray, typing.Callable]:

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):

        return _array_divide(a, b)

    elif callable(a) and callable(b):

        return _generator_factory_divide(a, b)

    else:

        raise ValueError("`a` and `b` must both be arrays or both be callable")

########################################################################################################################
