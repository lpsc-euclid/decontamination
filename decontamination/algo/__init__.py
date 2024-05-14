# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import typing

import numpy as np
import numba as nb
import numba.cuda as cu

from .. import jit

########################################################################################################################
# DATASET UTILITIES                                                                                                    #
########################################################################################################################

def dataset_to_generator_builder(dataset: typing.Union[tuple, np.ndarray, typing.Callable]) -> typing.Callable:

    return dataset if callable(dataset) else lambda: lambda: (dataset, )

########################################################################################################################

def batch_iterator(size: int, n_max_per_batch: typing.Optional[int]) -> typing.Generator[typing.Tuple[int, int], None, None]:

    ####################################################################################################################

    if n_max_per_batch is None:

        yield 0, size

        return

    ####################################################################################################################

    n_chunks, n_remaining = divmod(size, n_max_per_batch)

    ####################################################################################################################

    for i in range(n_chunks):

        s = i * n_max_per_batch
        e = s + n_max_per_batch

        yield s, e

    ####################################################################################################################

    if n_remaining > 0:

        yield n_max_per_batch * n_chunks, size

########################################################################################################################
# ASYMPTOTIC DECAY                                                                                                     #
########################################################################################################################

@nb.njit(inline = 'always')
def asymptotic_decay_cpu(epoch: int, epochs: int) -> float:

    return 1.0 / (1.0 + 2.0 * epoch / epochs)

########################################################################################################################

@cu.jit(inline = True)
def asymptotic_decay_gpu(epoch: int, epochs: int) -> float:

    return 1.0 / (1.0 + 2.0 * epoch / epochs)

########################################################################################################################
# CPU & GPU UTILITIES                                                                                                  #
########################################################################################################################

@jit(inline = True)
def atomic_add_vector_xpu(dest: np.ndarray, src: np.ndarray) -> None:

    for i in range(dest.shape[0]):

        jit.atomic_add(dest, i, src[i])

########################################################################################################################

@jit(inline = True)
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
