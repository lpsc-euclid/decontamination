# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb
import numba.cuda as cu

from .. import jit, device_array_empty, device_array_zeros

from . import dataset_to_generator_builder

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
            Topology of the model, either **'square'** or **'hexagonal'** (default: **None**, uses: **'hexagonal'**).
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

        self._weights = np.empty((self._m * self._n, self._dim), dtype = self._dtype)

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
    def _neuron_locations_square(m: int, n: int) -> typing.Iterator[typing.List[int]]:

        for i in range(m):
            for j in range(n):

                yield [i, j]

    ####################################################################################################################

    @staticmethod
    def _neuron_locations_hexagonal(m: int, n: int) -> typing.Iterator[typing.List[float]]:

        for i in range(m):
            for j in range(n):

                i_offset = (j & 1) * (-0.5)

                yield [i + i_offset, j * 0.8660254037844386]  # √3/2 = 0.8660254037844386

    ####################################################################################################################

    def _rebuild_topography(self):

        if self._topology == 'square':

            self._topography = np.array(list(SOM_Abstract._neuron_locations_square(self._m, self._n)), dtype = np.float32)

        else:

            self._topography = np.array(list(SOM_Abstract._neuron_locations_hexagonal(self._m, self._n)), dtype = np.float32)

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

            ############################################################################################################
            # HEADERS                                                                                                  #
            ############################################################################################################

            for name, field in header_extra.items():

                data = getattr(self, field)

                if data is not None:

                    model_group.attrs[name] = data

            ############################################################################################################
            # DATASETS                                                                                                 #
            ############################################################################################################

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

            ############################################################################################################
            # HEADERS                                                                                                  #
            ############################################################################################################

            for name, field in header_extra.items():

                try:

                    value = model_group.attrs[name]

                    if isinstance(value, np.int32) or isinstance(value, np.int64):

                        setattr(self, field, int(value))

                    elif isinstance(value, np.float32) or isinstance(value, np.float64):

                        setattr(self, field, float(value))

                    else:

                        setattr(self, field, value)

                except KeyError:

                    pass

            ############################################################################################################
            # DATASETS                                                                                                 #
            ############################################################################################################

            for name, field in dataset_extra.items():

                try:

                    array = np.array(model_group[name])

                    array = array.astype(self._dtype)

                    setattr(self, field, array)

                except KeyError:

                    pass

        ################################################################################################################

        self._rebuild_topography()

    ####################################################################################################################

    @property
    def m(self) -> int:

        """
        Returns the number of neuron rows.
        """

        return self._m

    ####################################################################################################################

    @property
    def n(self) -> int:

        """
        Returns the number of neuron columns.
        """

        return self._n

    ####################################################################################################################

    @property
    def dim(self) -> int:

        """
        Returns the dimensionality of the input data.
        """

        return self._dim

    ####################################################################################################################

    @property
    def dtype(self) -> typing.Type[np.single]:

        """
        Returns the neural network data type.
        """

        return self._dtype

    ####################################################################################################################

    @property
    def topology(self) -> str:

        """
        Returns the model topology, either **'square'** or **'hexagonal'**.
        """

        return self._topology

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
        Returns the quantization error. $$ c_i^1\\equiv\\mathrm{1^\\mathrm{st}\\,bmu}\\equiv\\underset{j}{\\mathrm{arg\\,min}_1}\\lVert x_i-w_j\\rVert $$ $$ \\boxed{e_Q\\equiv\\frac{1}{N}\\sum_{i=1}^N\\lVert x_i-w_{c_i^1}\\rVert} $$
        """

        return self._quantization_errors

    ####################################################################################################################

    def get_topographic_errors(self) -> np.ndarray:

        """
        Returns the topographic errors. $$ c_i^n\\equiv\\mathrm{n^\\mathrm{th}\\,bmu}\\equiv\\underset{j}{\\mathrm{arg\\,min}_n}\\lVert x_i-w_j\\rVert $$ $$ r\\equiv\\left\\{\\begin{array}{ll}\\sqrt{1}&\\mathrm{topology=hexagon}\\\\\\sqrt{2}&\\mathrm{topology=square}\\end{array}\\right. $$ $$ t(x_i)\\equiv\\left\\{\\begin{array}{ll}1&\\lVert c_i^1-c_i^2\\rVert>r\\\\0&\\mathrm{otherwise}\\end{array}\\right. $$ $$ \\boxed{e_t\\equiv\\frac{1}{N}\\sum_{i=0}^Nt(x_i)} $$
        """

        return self._topographic_errors

    ####################################################################################################################

    _X_HEX_STENCIL = np.array([
        +1, +1, +1, +0, -1, +0,  # Even line
        +0, +1, +0, -1, -1, -1,  # Odd line
    ], dtype = int)

    _Y_HEX_STENCIL = np.array([
        +1, +0, -1, -1, +0, +1,  # Even line
        +1, +0, -1, -1, +0, +1,  # Odd line
    ], dtype = int)

    ####################################################################################################################

    _X_SQU_STENCIL = np.array([
        +0, -1, -1, -1, +0, +1, +1, +1,  # Even line
        +0, -1, -1, -1, +0, +1, +1, +1,  # Odd line
    ], dtype = int)

    _Y_SQU_STENCIL = np.array([
        -1, -1, +0, +1, +1, +1, +0, -1,  # Even line
        -1, -1, +0, +1, +1, +1, +0, -1,  # Odd line
    ], dtype = int)

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _distance_map(result, centroids: np.ndarray, x_stencil: np.ndarray, y_stencil: np.ndarray, m: int, n: int, l: int) -> None:

        for x in range(m):
            for y in range(n):

                offset = (y & 1) * l

                w1 = centroids[x, y]

                for z in range(l):

                    i = x + x_stencil[z + offset]
                    j = y + y_stencil[z + offset]

                    if 0 <= i < m and 0 <= j < n:

                        w2 = centroids[i, j]

                        result[x, y, z] = np.sqrt(np.sum((w1 - w2) ** 2))

    ####################################################################################################################

    def get_distance_map(self, scaling: typing.Optional[str] = None) -> np.ndarray:

        """
        Returns a matrix of the distances between each weight.

        Parameters
        ----------
        scaling : typing.Optional[str]
            Normalization method, either '**sum**' or '**mean**' (default: **None**, uses: '**sum**').
        """

        scaling = scaling or 'sum'

        ################################################################################################################

        if self._topology == 'square':

            result = np.full((self._m, self._n, 8), np.nan, dtype = self._dtype)

            SOM_Abstract._distance_map(result, self.get_centroids(), SOM_Abstract._X_SQU_STENCIL, SOM_Abstract._Y_SQU_STENCIL, self._m, self._n, 8)

        else:

            result = np.full((self._m, self._n, 6), np.nan, dtype = self._dtype)

            SOM_Abstract._distance_map(result, self.get_centroids(), SOM_Abstract._X_HEX_STENCIL, SOM_Abstract._Y_HEX_STENCIL, self._m, self._n, 6)

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

    def get_activation_map(self, dataset: typing.Union[np.ndarray, typing.Callable], enable_gpu: bool = True, threads_per_blocks: int = 1024) -> np.ndarray:

        """
        Returns a matrix containing the number of times each neuron have been activated for the given input.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Dataset array or generator builder.
        enable_gpu : bool
            If available, run on GPU rather than CPU (default: **True**).
        threads_per_blocks : int
            Number of GPU threads per blocks (default: **1024**).
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        result = device_array_zeros(shape = (self._m * self._n, ), dtype = np.int64)

        ################################################################################################################

        generator = generator_builder()

        for data in generator():

            _count_bmus_kernel[enable_gpu, threads_per_blocks, data.shape[0]](result, self._weights, data, self._m * self._n)

        ################################################################################################################

        return result.copy_to_host().reshape((self._m, self._n))

    ####################################################################################################################

    def get_winners(self, dataset: np.ndarray, enable_gpu: bool = True, threads_per_blocks: int = 1024) -> np.ndarray:

        """
        Returns a vector of the best matching unit indices (shape m×n) for the given input.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset array.
        enable_gpu : bool
            If available, run on GPU rather than CPU (default: **True**).
        threads_per_blocks : int
            Number of GPU threads per blocks (default: **1024**).
        """

        ################################################################################################################

        result = device_array_empty(dataset.shape[0], dtype = np.int32)

        ################################################################################################################

        _find_bmus_kernel[enable_gpu, threads_per_blocks, dataset.shape[0]](result, self._weights, dataset, self._m * self._n)

        ################################################################################################################

        return result.copy_to_host()

########################################################################################################################

@jit(kernel = True, parallel = False)
def _count_bmus_kernel(result: np.ndarray, weights: np.ndarray, vectors: np.ndarray, mn: int) -> None:

    ####################################################################################################################
    # !--BEGIN-CPU--

    for i in nb.prange(vectors.shape[0]):

        result[_find_bmu_xpu(weights, vectors[i], mn)] += 1

    # !--END-CPU--
    ####################################################################################################################
    # !--BEGIN-GPU--

    i = cu.grid(1)

    if i < vectors.shape[0]:

        # noinspection PyArgumentList
        cu.atomic.add(result, _find_bmu_xpu(weights, vectors[i], mn), 1)

    # !--END-GPU--

########################################################################################################################

@jit(kernel = True, parallel = True)
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
