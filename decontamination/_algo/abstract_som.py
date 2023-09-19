# -*- coding: utf-8 -*-
########################################################################################################################

import abc
import typing

import numpy as np
import numba as nb

########################################################################################################################

class AbstractSOM(abc.ABC):

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: np.dtype = np.float32, topology: typing.Optional[str] = None):

        """
        Constructor for the Abstract Self Organizing Map (SOM).

        Parameters
        ----------
        m : int
            Number of neuron rows.
        n : int
            Number of neuron columns.
        dim : int
            Dimensionality of the input data.
        dtype : np.dtype
            Neural network data type (default: **np.float32**).
        topology : Optional[str]
            Topology of the map, '**square**' or '**hexagonal**' (default: '**hexagonal**').
        """

        ################################################################################################################

        self._m = m
        self._n = n
        self._dim = dim
        self._dtype = dtype
        self._topology = topology or 'hexagonal'

        ################################################################################################################

        self._weights = np.empty(shape = (self._m * self._n, self._dim), dtype = self._dtype)

    ####################################################################################################################

    def init_from(self, other: 'AbstractSOM') -> None:

        """
        Initializes the neural network from another one.

        Parameters
        ----------
        other : AbstractSOM
            Another SOM object from which the weights will be copied.
        """

        if self._m != other._m        \
           or                         \
           self._n != other._n        \
           or                         \
           self._dim != other._dim    \
           or                         \
           self._dtype != other._dtype:

            raise Exception('Incompatible shapes or dtypes')

        self._weights[:] = other._weights[:]

    ####################################################################################################################

    def get_weights(self) -> np.ndarray:

        """
        Returns the neural network weights with the shape: [m * n, dim].
        """

        return self._weights.reshape((self._m * self._n, self._dim))

    ####################################################################################################################

    def get_centroids(self) -> np.ndarray:

        """
        Returns of the neural network weights with the shape: [m, n, dim].
        """

        return self._weights.reshape((self._m, self._n, self._dim))

    ####################################################################################################################

    _X_HEX_STENCIL = np.array([
        +1, +1, +1,  0, -1,  0,  # Even line
         0, +1,  0, -1, -1, -1,  # Odd line
    ], dtype = np.int64)

    _Y_HEX_STENCIL = np.array([
        +1, 0, -1, -1, 0, +1,  # Even line
        +1, 0, -1, -1, 0, +1,  # Odd line
    ], dtype = np.int64)

    ####################################################################################################################

    _X_SQU_STENCIL = np.array([
        0, -1, -1, -1, 0, +1, +1, +1,  # Even line
        0, -1, -1, -1, 0, +1, +1, +1,  # Odd line
    ], dtype = np.int64)

    _Y_SQU_STENCIL = np.array([
        -1, -1, 0, +1, +1, +1, 0, -1,  # Even line
        -1, -1, 0, +1, +1, +1, 0, -1,  # Odd line
    ], dtype = np.int64)

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _distance_map_kernel(result, centroids: np.ndarray, x_stencil: np.ndarray, y_stencil: np.ndarray, m: int, n: int, l: int) -> None:

        for x in range(m):
            for y in range(n):

                offset = (y & 1) * l

                w = centroids[x, y]

                for k in range(8):

                    i = x + x_stencil[k + offset]
                    j = y + y_stencil[k + offset]

                    if 0 <= i < m and 0 <= j < n:

                        diff = w - centroids[i, j]

                        result[x, y, k] = np.sqrt(np.sum(diff * diff))

    ####################################################################################################################

    def distance_map(self, scaling: typing.Optional[str] = None) -> np.ndarray:

        """
        Returns the distance map of the neural network weights.

        Parameters
        ----------
        scaling : Optional[str]
            Normalization method, '**sum**' or '**mean**' (default: '**sum**')
        """

        scaling = scaling or 'sum'

        ################################################################################################################

        if self._topology == 'hexagonal':

            result = np.full(shape = (self._m, self._n, 6), fill_value = np.nan, dtype = self._dtype)

            AbstractSOM._distance_map_kernel(result, self.get_centroids(), AbstractSOM._X_HEX_STENCIL, AbstractSOM._Y_HEX_STENCIL, self._m, self._n, 6)

        else:

            result = np.full(shape = (self._m, self._n, 8), fill_value = np.nan, dtype = self._dtype)

            AbstractSOM._distance_map_kernel(result, self.get_centroids(), AbstractSOM._X_SQU_STENCIL, AbstractSOM._Y_SQU_STENCIL, self._m, self._n, 8)

        ################################################################################################################

        if scaling == 'sum':
            result = np.nansum(result, axis = 2)

        elif scaling == 'mean':
            result = np.nanmean(result, axis = 2)

        else:
            raise Exception(f'Invalid scaling method `{scaling}`')

        ################################################################################################################

        return result / result.max()

########################################################################################################################
