# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb
import numba.cuda as cu

from .. import jit, result_array

from . import batch_iterator, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class SOM_Abstract(object):

    """
    Self Organizing Maps (abstract class).
    """

    __MODE__ = 'abstract'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[np.single] = np.float32, topology: typing.Optional[str] = None):

        """
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

        self._quantization_errors = np.empty(0, dtype = np.float32)

        self._topographic_errors = np.empty(0, dtype = np.float32)

        ################################################################################################################

        self._header_extra = {
            'mode': '__MODE__',
        }

        self._dataset_extra = {
        }

    ####################################################################################################################

    @staticmethod
    def _neuron_locations(m: int, n: int) -> typing.Iterator[typing.List[int]]:

        for i in range(m):

            for j in range(n):

                yield [i, j]

    ####################################################################################################################

    def _rebuild_topography(self):

        self._topography = np.array(list(SOM_Abstract._neuron_locations(self._m, self._m)), dtype = np.int64)

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

    def init_from(self, other: 'SOM_Abstract') -> None:

        """
        Initializes the neural network from another one.

        Parameters
        ----------
        other : SOM_Abstract
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

    def _init_hdf5_extra(self):

        ################################################################################################################

        header_extra = {name: field for name, field in self._header_extra.items()}

        dataset_extra = {name: field for name, field in self._dataset_extra.items()}

        ################################################################################################################

        header_extra['m'] = '_m'
        header_extra['n'] = '_n'
        header_extra['dim'] = '_dim'
        header_extra['topology'] = '_topology'

        dataset_extra['weights'] = '_weights'
        dataset_extra['quantization_errors'] = '_quantization_errors'
        dataset_extra['topographic_errors'] = '_topographic_errors'

        ################################################################################################################

        return header_extra, dataset_extra

    ####################################################################################################################

    def save(self, filename: str) -> None:

        """
        Saves the trained neural network to a HDF5 file.

        Parameters
        ----------
        filename : str
            Output filename.
        """

        ################################################################################################################

        import h5py

        ################################################################################################################

        header_extra, dataset_extra = self._init_hdf5_extra()

        ################################################################################################################

        with h5py.File(filename, mode = 'w') as file:

            model_group = file.create_group('model', track_order = True)

            # HEADERS #

            for name, field in header_extra.items():

                data = getattr(self, field)

                if data is not None:

                    model_group.attrs[name] = data

            # DATASETS #

            for name, field in dataset_extra.items():

                data = getattr(self, field)

                if data is not None:

                    model_group.create_dataset(
                        name,
                        data = data,
                        shape = data.shape,
                        dtype = data.dtype
                    )

        ################################################################################################################

        self._rebuild_topography()

    ####################################################################################################################

    def load(self, filename: str) -> None:

        """
        Loads the trained neural network from a HDF5 file.

        Parameters
        ----------
        filename : str
            Input filename.
        """

        ################################################################################################################

        import h5py

        ################################################################################################################

        header_extra, dataset_extra = self._init_hdf5_extra()

        ################################################################################################################

        with h5py.File(filename, mode = 'r') as file:

            model_group = file['model']

            # HEADERS #

            for name, field in header_extra.items():

                try:
                    setattr(self, field, np.array(model_group.attrs[name]))
                except KeyError:
                    pass

            # DATASETS #

            for name, field in dataset_extra.items():

                try:
                    setattr(self, field, np.array(model_group[name]))
                except KeyError:
                    pass

        ################################################################################################################

        self._rebuild_topography()

    ####################################################################################################################

    def get_weights(self) -> np.ndarray:

        """
        Returns the neural network weights with the shape [m * n, dim].
        """

        return self._weights.reshape((self._m * self._n, self._dim))

    ####################################################################################################################

    def get_centroids(self) -> np.ndarray:

        """
        Returns the neural network weights with the shape [m, n, dim].
        """

        return self._weights.reshape((self._m, self._n, self._dim))

    ####################################################################################################################

    def get_quantization_errors(self) -> np.ndarray:

        """
        Returns the quantization error. $$ c_i^1=\\mathrm{1^\\mathrm{st}\\,bmu}=\\underset{j}{\\mathrm{arg\\,min}_1}\\lVert x_i-w_j\\rVert $$ $$ \\boxed{e_Q=\\frac{1}{N}\\sum_{i=1}^N\\lVert x_i-w_{c_i^1}\\rVert} $$
        """

        return self._quantization_errors

    ####################################################################################################################

    def get_topographic_errors(self) -> np.ndarray:

        """
        Returns the topographic errors. $$ c_i^n=\\mathrm{n^\\mathrm{th}\\,bmu}=\\underset{j}{\\mathrm{arg\\,min}_n}\\lVert x_i-w_j\\rVert $$ $$ t(x_i)=\\left\\{\\begin{array}{ll}1&\\lVert c_i^1-c_i^2\\rVert>\\sqrt{2}\\\\0&\\mathrm{otherwise}\\end{array}\\right. $$ $$ \\boxed{e_t=\\frac{1}{N}\\sum_{i=0}^Nt(x_i)} $$
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
    def _distance_map(result, centroids: np.ndarray, x_stencil: np.ndarray, y_stencil: np.ndarray, m: int, n: int, l: int) -> None:

        for x in range(m):
            for y in range(n):

                offset = (y & 1) * l

                w = centroids[x, y]

                for z in range(l):

                    i = x + x_stencil[z + offset]
                    j = y + y_stencil[z + offset]

                    if 0 <= i < m and 0 <= j < n:

                        diff = w - centroids[i, j]

                        result[x, y, z] = np.sqrt(np.sum(diff ** 2))

    ####################################################################################################################

    def get_distance_map(self, scaling: typing.Optional[str] = None) -> np.ndarray:

        """
        Returns the distance map of the neural network weights.

        Parameters
        ----------
        scaling : typing.Optional[str]
            Normalization method, either '**sum**' or '**mean**' (default: '**sum**').
        """

        scaling = scaling or 'sum'

        ################################################################################################################

        if self._topology == 'hexagonal':

            result = np.full(shape = (self._m, self._n, 6), fill_value = np.nan, dtype = self._dtype)

            SOM_Abstract._distance_map(result, self.get_centroids(), SOM_Abstract._X_HEX_STENCIL, SOM_Abstract._Y_HEX_STENCIL, self._m, self._n, 6)

        else:

            result = np.full(shape = (self._m, self._n, 8), fill_value = np.nan, dtype = self._dtype)

            SOM_Abstract._distance_map(result, self.get_centroids(), SOM_Abstract._X_SQU_STENCIL, SOM_Abstract._Y_SQU_STENCIL, self._m, self._n, 8)

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

    def get_activation_map(self, dataset: typing.Union[np.ndarray, typing.Callable], n_chunks: int = 1, enable_gpu: bool = True, threads_per_blocks: typing.Union[typing.Tuple[int], int] = 1024) -> np.ndarray:

        """
        ???

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            ???
        n_chunks : int
            ???
        enable_gpu : bool
            ???
        threads_per_blocks : typing.Union[typing.Tuple[int], int]
            ???
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        generator = generator_builder()

        ################################################################################################################

        result = np.zeros(self._m * self._n, dtype = np.int64)

        ################################################################################################################

        for data in generator():

            for chunk in batch_iterator(data, n_chunks):

                bmus = result_array(data.shape[0], dtype = np.int32)

                _find_bmus_kernel[enable_gpu, threads_per_blocks, data.shape[0]](bmus, self._weights, chunk, self._m * self._n)

                SOM_Abstract._count_bmus(result, bmus.copy_to_host())

        ################################################################################################################

        return result.reshape((self._m, self._n, ))

    ####################################################################################################################

    def get_winners(self, dataset: np.ndarray, locations: bool = False, enable_gpu: bool = True, threads_per_blocks: typing.Union[typing.Tuple[int], int] = 1024) -> np.ndarray:

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

        bmus = result_array(dataset.shape[0], dtype = np.int32)

        _find_bmus_kernel[enable_gpu, threads_per_blocks, dataset.shape[0]](bmus, self._weights, dataset, self._m * self._n)

        result = bmus.copy_to_host()

        ################################################################################################################

        return self._topography[result] if locations else result

########################################################################################################################

@jit(kernel = True)
def _find_bmus_kernel(result: np.ndarray, weights: np.ndarray, vectors: np.ndarray, mn: int) -> None:

    ####################################################################################################################
    # !--BEGIN-CPU--

    for i in nb.prange(vectors.shape[0]):

        result[i] = _find_bmu_xpu(weights, vectors[i], mn)

    # !--END-CPU--
    ####################################################################################################################
    # !--BEGIN-GPU--

    i = cu.grid(1)
    if i < vectors.shape[0]:

        result[i] = _find_bmu_xpu(weights, vectors[i], mn)

    # !--END-GPU--

########################################################################################################################

@jit(parallel = False)
def _find_bmu_xpu(weights: np.ndarray, vector: np.ndarray, mn: int) -> int:

    min_distance = 1.0e99
    min_index = 0x00

    for index in range(mn):

        ################################################################################################################
        # !--BEGIN-CPU--

        distance = np.sum((weights[index] - vector) ** 2)

        # !--END-CPU--
        ################################################################################################################
        # !--BEGIN-GPU--

        distance = 0.0

        weight = weights[index]

        for i in range(vector.shape[0]):

            distance += (weight[i] - vector[i]) ** 2

        # !--END-GPU--
        ################################################################################################################

        if min_distance > distance:

            min_distance = distance
            min_index = index

        ################################################################################################################

    return min_index

########################################################################################################################
