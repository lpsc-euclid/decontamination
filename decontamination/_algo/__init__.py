# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb
import numba.cuda as cu

from .. import jit

########################################################################################################################
# DATASET UTILITIES                                                                                                    #
########################################################################################################################

def dataset_to_generator_builder(dataset: typing.Union[np.ndarray, typing.Callable]) -> typing.Callable:

    return dataset if callable(dataset) else lambda: lambda: (dataset, )

########################################################################################################################

def batch_iterator(vectors: np.ndarray, n_chunks: int) -> typing.Iterator[np.ndarray]:

    ####################################################################################################################

    chunk_size, chunk_remaining = divmod(vectors.shape[0], n_chunks)

    ####################################################################################################################

    for i in range(n_chunks):

        s = i * chunk_size
        e = s + chunk_size

        yield vectors[s: e]

    ####################################################################################################################

    if chunk_remaining > 0:

        yield vectors[n_chunks * chunk_size:]

########################################################################################################################

@nb.njit(inline = 'always')
def asymptotic_decay(epoch: int, epochs: int) -> float:

    return 1.0 / (1.0 + 2.0 * epoch / epochs)

########################################################################################################################

@jit(parallel = False)
def asymptotic_decay_xpu(epoch: int, epochs: int) -> float:

    return 1.0 / (1.0 + 2.0 * epoch / epochs)

########################################################################################################################
# CPU & GPU UTILITIES                                                                                                  #
########################################################################################################################

@jit(parallel = False)
def atomic_add_vector_xpu(dest: np.ndarray, src: np.ndarray) -> None:

    for i in range(dest.shape[0]):

        jit.atomic_add(dest, i, src[i])

########################################################################################################################

@jit(parallel = False)
def square_distance_xpu(vector1: np.ndarray, vector2: np.ndarray) -> float:

    ####################################################################################################################
    # !--BEGIN-CPU--

    return np.sum((vector1 - vector2) ** 2, axis = -1)

    # !--END-CPU--
    ####################################################################################################################
    # !--BEGIN-GPU--

    result = 0.0

    for i in range(vector1.shape[0]):

        result += (vector1[i] - vector2[i]) ** 2

    return result

    # !--END-GPU--

########################################################################################################################
