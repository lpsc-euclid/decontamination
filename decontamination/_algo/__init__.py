# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numba as nb
import numpy as np

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

        yield vectors[n_chunks * chunk_size: ]

########################################################################################################################

@nb.njit(parallel = False)
def asymptotic_decay(epoch: int, epochs: int) -> float:

    return 1.0 / (1.0 + 2.0 * epoch / epochs)

########################################################################################################################
