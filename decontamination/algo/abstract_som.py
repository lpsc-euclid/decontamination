# -*- coding: utf-8 -*-
########################################################################################################################

import abc

import numpy as np
import numba as nb

########################################################################################################################

class AbstractSOM(abc.ABC):

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: type = np.float32):

        """
        Constructor for the Abstract Self Organizing Map (SOM).

        Arguments
        ---------
            m : int
                Number of neuron rows.
            n : int
                Number of neuron columns.
            dim : int
                Dimensionality of the input data.
            dtype : type
                Neural network data type (default: **np.float32**).
        """

        ################################################################################################################

        self._m = m
        self._n = n
        self._dim = dim
        self._dtype = dtype

        ################################################################################################################

        self._weights = np.empty(shape = (self._m * self._n, self._dim), dtype = self._dtype)

    ####################################################################################################################

    def init_from(self, other: 'AbstractSOM') -> None:

        """
        Initializes the neural network from an other one.

        Arguments
        ---------
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
        Returns the neural network weights (shape = [m * n, dim]).
        """

        return self._weights.reshape((self._m * self._n, self._dim))

    ####################################################################################################################

    def get_centroids(self) -> np.ndarray:

        """
        Returns of the neural network weights (shape = [m, n, dim]).
        """

        return self._weights.reshape((self._m, self._n, self._dim))

    ####################################################################################################################

    II = np.array([
        0, -1, -1, -1, 0, 1, 1, 1,
        0, -1, -1, -1, 0, 1, 1, 1,
    ], dtype = np.int64)

    JJ = np.array([
        -1, -1, 0, 1, 1, 1, 0, -1,
        -1, -1, 0, 1, 1, 1, 0, -1,
    ], dtype = np.int64)

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _distance_map_kernel(result, centroids, ii, jj, m, n):

        ################################################################################################################

        for x in range(m):
            for y in range(n):

                w = centroids[x, y]

                offset = 8 if y % 2 == 0 else 0

                for k in range(8):

                    i = x + ii[k + offset]
                    j = y + jj[k + offset]

                    if 0 <= i < m and 0 <= j < n:

                        diff = w - centroids[i, j]

                        result[x, y, k] = np.sqrt(np.sum(diff, diff))

        ################################################################################################################

        result = np.nansum(result, axis = 2)

        result /= result.max()

    ####################################################################################################################

    def distance_map(self) -> np.ndarray:

        """
        Returns the distance map of the neural network weights.
        """

        result = np.full(shape = (self._m, self._n, 8), fill_value = np.nan, dtype = self._dtype)

        AbstractSOM._distance_map_kernel(result, self.get_centroids(), AbstractSOM.II, AbstractSOM.JJ, self._m, self._n)

        return result

########################################################################################################################
