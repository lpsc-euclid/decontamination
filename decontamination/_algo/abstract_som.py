# -*- coding: utf-8 -*-
########################################################################################################################

import abc
import typing

import numpy as np
import numba as nb

########################################################################################################################

class AbstractSOM(abc.ABC):

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: np.dtype = np.float32, topology: typing.Optional[str] = None, seed: int = None):

        """
        Constructor for an Abstract Self Organizing Map (SOM).

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
        topology : typing.Optional[str]
            Topology of the map, either '**square**' or '**hexagonal**' (default: '**hexagonal**').
        seed : int
            Seed for random generator (default: **None**).
        """

        ################################################################################################################

        self._m = m
        self._n = n
        self._dim = dim
        self._dtype = dtype
        self._topology = topology or 'hexagonal'
        self._seed = seed

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

        if self._m != other._m              \
           or                               \
           self._n != other._n              \
           or                               \
           self._dim != other._dim          \
           or                               \
           self._dtype != other._dtype      \
           or                               \
           self._topology != other._topology:

            raise Exception('Incompatible shapes, dtypes or topologies')

        self._weights[:] = other._weights[:]

    ####################################################################################################################

    def save(
        self,
        filename: str,
        header_extra: typing.Optional[typing.Dict[str, str]] = None,
        dataset_extra: typing.Optional[typing.Dict[str, str]] = None
    ) -> None:

        """
        Saves the trained neural network to a file.

        Parameters
        ----------
        filename : str
            Output HDF5 filename.
        header_extra : typing.Optional[typing.Dict[str, str]]
            Dictionary of extra headers (name, field name in class).
        dataset_extra : typing.Optional[typing.Dict[str, str]]
            Dictionary of extra datasets (name, field name in class).
        """

        ################################################################################################################

        import h5py

        if header_extra is None:
            header_extra = {}

        if dataset_extra is None:
            dataset_extra = {}

        ################################################################################################################

        header_extra['m'] = '_m'
        header_extra['n'] = '_n'
        header_extra['dim'] = '_dim'
        header_extra['topology'] = '_topology'

        dataset_extra['weights'] = '_weight'

        ################################################################################################################

        with h5py.File(filename, 'w') as file:

            for name, field in header_extra.items():

                file.attrs[name] = getattr(self, field)

            for name, field in dataset_extra.items():

                data = getattr(self, field)

                file.create_dataset(
                    name,
                    data = data,
                    shape = data.shape,
                    dtype = data.dtype
                )

    ####################################################################################################################

    def load(
        self,
        filename: str,
        header_extra: typing.Optional[typing.Dict[str, str]] = None,
        dataset_extra: typing.Optional[typing.Dict[str, str]] = None
    ) -> None:

        """
        Loads the trained neural network from a file.

        Parameters
        ----------
        filename : str
            Input HDF5 filename.
        header_extra : typing.Optional[typing.Dict[str, str]]
            Dictionary of extra headers (name, field name in class).
        dataset_extra : typing.Optional[typing.Dict[str, str]]
            Dictionary of extra datasets (name, field name in class).
        """

        ################################################################################################################

        import h5py

        if header_extra is None:
            header_extra = {}

        if dataset_extra is None:
            dataset_extra = {}

        ################################################################################################################

        header_extra['m'] = '_m'
        header_extra['n'] = '_n'
        header_extra['dim'] = '_dim'
        header_extra['topology'] = '_topology'

        dataset_extra['weights'] = '_weight'

        ################################################################################################################

        with h5py.File(filename, 'r') as file:

            for name, field in header_extra.items():

                setattr(self, field, file.attrs[name])

            for name, field in dataset_extra.items():

                setattr(self, field, file[name])

    ####################################################################################################################

    def get_weights(self) -> np.ndarray:

        """
        Returns the neural network weights with the shape: [m * n, dim].
        """

        return self._weights.reshape((self._m * self._n, self._dim))

    ####################################################################################################################

    def get_centroids(self) -> np.ndarray:

        """
        Returns the neural network weights with the shape: [m, n, dim].
        """

        return self._weights.reshape((self._m, self._n, self._dim))

    ####################################################################################################################

    _X_HEX_STENCIL = np.array([
        +1, +1, +1, +0, -1, +0,  # Even line
        +0, +1, +0, -1, -1, -1,  # Odd line
    ], dtype = np.int64)

    _Y_HEX_STENCIL = np.array([
        +1, +0, -1, -1, +0, +1,  # Even line
        +1, +0, -1, -1, +0, +1,  # Odd line
    ], dtype = np.int64)

    ####################################################################################################################

    _X_SQU_STENCIL = np.array([
        +0, -1, -1, -1, +0, +1, +1, +1,  # Even line
        +0, -1, -1, -1, +0, +1, +1, +1,  # Odd line
    ], dtype = np.int64)

    _Y_SQU_STENCIL = np.array([
        -1, -1, +0, +1, +1, +1, +0, -1,  # Even line
        -1, -1, +0, +1, +1, +1, +0, -1,  # Odd line
    ], dtype = np.int64)

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _distance_map_kernel(result, centroids: np.ndarray, x_stencil: np.ndarray, y_stencil: np.ndarray, m: int, n: int, l: int) -> None:

        for x in range(m):
            for y in range(n):

                offset = (y & 1) * l

                w = centroids[x, y]

                for k in range(l):

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
        scaling : typing.Optional[str]
            Normalization method, either '**sum**' or '**mean**' (default: '**sum**')
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
