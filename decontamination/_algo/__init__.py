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

    size = len(vectors)

    chunk_size = size // n_chunks

    for i in range(n_chunks):

        s = i * chunk_size #############################
        e = s + chunk_size if i < n_chunks - 1 else size

        yield vectors[s: e]

########################################################################################################################

@nb.njit(parallel = False)
def asymptotic_decay(epoch: int, epochs: int) -> float:

    return 1.0 / (1.0 + 2.0 * epoch / epochs)

########################################################################################################################
