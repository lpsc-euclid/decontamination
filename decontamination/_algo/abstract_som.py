# -*- coding: utf-8 -*-
########################################################################################################################

import abc
import tqdm
import typing

import numpy as np
import numba as nb
import numba.cuda as cu

from .. import jit, GPU_OPTIMIZATION_AVAILABLE

from . import dataset_to_generator_builder

########################################################################################################################

class AbstractSOM(abc.ABC):

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[np.single] = np.float32, topology: typing.Optional[str] = None):

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
        dtype : typing.Type[np.single]
            Neural network data type (default: **np.float32**).
        topology : typing.Optional[str]
            Topology of the map, either '**square**' or '**hexagonal**' (default: '**hexagonal**').
        """

        ################################################################################################################

        self._m = m
        self._n = n
        self._dim = dim
        self._dtype = dtype
        self._topology = topology or 'hexagonal'

        ################################################################################################################

        self._rebuild_topography()

        ################################################################################################################

        self._weights = np.empty(shape = (self._m * self._n, self._dim), dtype = self._dtype)

        ################################################################################################################

        self._quantization_errors = np.empty(0, dtype = np.float32)
        self._topographic_errors = np.empty(0, dtype = np.float32)

    ####################################################################################################################

    @staticmethod
    def _neuron_locations(m: int, n: int) -> typing.Iterator[typing.List[int]]:

        for i in range(m):

            for j in range(n):

                yield [i, j]

    ####################################################################################################################

    def _rebuild_topography(self):

        self._topography = np.array(list(AbstractSOM._neuron_locations(self._m, self._m)), dtype = np.int64)

    ####################################################################################################################

    def init_rand(self, seed: typing.Optional[int] = None) -> None:

        """
        Initializes the neural network randomly.

        Parameters
        ----------
        seed : typing.Optional[int]
            Seed for random generator (default: **None**).
        """

        ################################################################################################################

        if seed is None:

            rng = np.random.default_rng()

        else:

            rng = np.random.default_rng(seed = seed)

        ################################################################################################################

        self._weights[...] = rng.random((self._m * self._n, self._dim), dtype = self._dtype)

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

        self._weights[...] = other._weights

    ####################################################################################################################

    @staticmethod
    def _init_hdf5_extra(header_extra, dataset_extra):

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

        return header_extra, dataset_extra

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

        ################################################################################################################

        header_extra, dataset_extra = AbstractSOM._init_hdf5_extra(header_extra, dataset_extra)

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

        ################################################################################################################

        header_extra, dataset_extra = AbstractSOM._init_hdf5_extra(header_extra, dataset_extra)

        ################################################################################################################

        with h5py.File(filename, 'r') as file:

            for name, field in header_extra.items():

                setattr(self, field, file.attrs[name])

            for name, field in dataset_extra.items():

                setattr(self, field, file[name])

        ################################################################################################################

        self._rebuild_topography()

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

    def get_quantization_errors(self) -> np.ndarray:

        """
        Returns the quantization error.
        """

        return self._quantization_errors

    ####################################################################################################################

    def get_topographic_errors(self) -> np.ndarray:

        """
        Returns the topographic errors.
        """

        return self._topographic_errors

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

                for z in range(l):

                    i = x + x_stencil[z + offset]
                    j = y + y_stencil[z + offset]

                    if 0 <= i < m and 0 <= j < n:

                        diff = w - centroids[i, j]

                        result[x, y, z] = np.sqrt(np.sum(diff * diff))

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

        return result / np.max(result)

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _count_bmus(result: np.ndarray, bmus: np.ndarray):

        for i in range(bmus.shape[0]):

            result[bmus[i]] += 1

    ####################################################################################################################

    def activation_map(self, dataset: typing.Union[np.ndarray, typing.Callable], enable_gpu: bool = True, threads_per_blocks: typing.Union[typing.Tuple[int], int] = 1024, show_progress_bar: bool = False) -> np.ndarray:

        """
        ???

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            ???
        enable_gpu : bool
            ???
        threads_per_blocks : typing.Union[typing.Tuple[int], int]
            ???
        show_progress_bar : bool
            Specifies whether to display a progress bar (default: **False**).
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        generator = generator_builder()

        ################################################################################################################

        result = np.zeros(self._m * self._n, dtype = np.int64)

        ################################################################################################################

        for data in tqdm.tqdm(generator(), disable = not show_progress_bar):

            if enable_gpu and GPU_OPTIMIZATION_AVAILABLE:

                bmus = cu.device_array(data.shape[0], dtype = np.int32)
                # noinspection PyUnresolvedReferences
                _find_bmus_kernel_gpu[threads_per_blocks, data.shape[0]](bmus, self._weights, data, self._m * self._n)
                AbstractSOM._count_bmus(result, bmus.copy_to_host())

            else:

                bmus = np.empty(data.shape[0], dtype = np.int32)
                # noinspection PyUnresolvedReferences
                _find_bmus_kernel_cpu[threads_per_blocks, data.shape[0]](bmus, self._weights, data, self._m * self._n)
                AbstractSOM._count_bmus(result, bmus)

        ################################################################################################################

        return result.reshape((self._m, self._n, ))

    ####################################################################################################################

    def winners(self, dataset: np.ndarray, locations: bool = False, enable_gpu: bool = True, threads_per_blocks: typing.Union[typing.Tuple[int], int] = 1024) -> np.ndarray:

        """
        ???

        Parameters
        ----------
        dataset : np.ndarray
            ???
        locations : bool
            ???
        enable_gpu : bool
            ???
        threads_per_blocks : typing.Union[typing.Tuple[int], int]
            ???
        """

        ################################################################################################################

        if enable_gpu and GPU_OPTIMIZATION_AVAILABLE:

            bmus = cu.device_array(dataset.shape[0], dtype = np.int32)
            # noinspection PyUnresolvedReferences
            _find_bmus_kernel_gpu[threads_per_blocks, dataset.shape[0]](bmus, self._weights, dataset, self._m * self._n)
            result = bmus.copy_to_host()

        else:

            bmus = np.empty(dataset.shape[0], dtype = np.int32)
            # noinspection PyUnresolvedReferences
            _find_bmus_kernel_cpu[threads_per_blocks, dataset.shape[0]](bmus, self._weights, dataset, self._m * self._n)
            result = bmus

        ################################################################################################################

        return self._topography[result] if locations else result

########################################################################################################################

@jit(kernel = True)
def _find_bmus_kernel_xpu(result: np.ndarray, weights: np.ndarray, vectors: np.ndarray, mn: int) -> None:

    ####################################################################################################################
    # !--BEGIN-CPU--

    for i in nb.prange(vectors.shape[0]):

        # noinspection PyUnresolvedReferences
        result[i] = _find_bmu_cpu(weights, vectors[i], mn)

    # !--END-CPU--
    ####################################################################################################################
    # !--BEGIN-GPU--

    i = cu.grid(1)
    if i < vectors.shape[0]:

        # noinspection PyUnresolvedReferences
        result[i] = _find_bmu_gpu(weights, vectors[i], mn)

    # !--END-GPU--

########################################################################################################################

@jit(parallel = False)
def _find_bmu_xpu(weights: np.ndarray, vector: np.ndarray, mn: int) -> int:

    min_dist_2 = 1.0e99
    min_index = 0x00

    for index in range(mn):

        ################################################################################################################
        # !--BEGIN-CPU--

        dist = np.sum((weights[index] - vector) ** 2)

        # !--END-CPU--
        ################################################################################################################
        # !--BEGIN-GPU--

        dist = 0.0

        weight = weights[index]

        for i in range(vector.shape[0]):

            dist += (weight[i] - vector[i]) ** 2

        # !--END-GPU--
        ################################################################################################################

        if min_dist_2 > dist:

            min_dist_2 = dist
            min_index = index

        ################################################################################################################

    return min_index

########################################################################################################################
