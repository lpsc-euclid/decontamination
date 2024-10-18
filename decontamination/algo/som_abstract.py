# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import math
import tqdm
import typing

import numpy as np
import numba as nb

from .. import jit, device_array_empty, device_array_zeros

from . import square_distance_xpu, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming, DuplicatedCode
class SOM_Abstract(object):

    """
    Self Organizing Maps (abstract class).

    Parameters
    ----------
    m : int
        Number of neuron rows.
    n : int
        Number of neuron columns.
    dim : int
        Dimensionality of the input data.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]], default: **np.float32**
        Neural network data type, either **np.float32** or **np.float64**.
    topology : str, default: **None** ≡ **'hexagonal'**
        Neural network topology, either **'square'** or **'hexagonal'**.
    """

    __MODE__ = 'abstract'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, topology: typing.Optional[str] = None):

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

                i_offset = (j & 1) * 0.5

                yield [i + i_offset, j * 0.8660254037844386]  # √3/2 = 0.8660254037844386

    ####################################################################################################################

    def _rebuild_topography(self):

        if self._topology == 'square':

            self._topography = np.array(list(SOM_Abstract._neuron_locations_square(self._m, self._n)), dtype = self._dtype)

        else:

            self._topography = np.array(list(SOM_Abstract._neuron_locations_hexagonal(self._m, self._n)), dtype = self._dtype)

    ####################################################################################################################

    def init_rand(self, seed: typing.Optional[int] = None) -> None:

        """
        Initializes the neural network randomly.

        Parameters
        ----------
        seed : int, default: **None**
            Optional seed for random generator.
        """

        ################################################################################################################

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

            raise TypeError('Incompatible shapes, dtypes or topologies')

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

        """Number of neuron rows."""

        return self._m

    ####################################################################################################################

    @property
    def n(self) -> int:

        """Number of neuron columns."""

        return self._n

    ####################################################################################################################

    @property
    def dim(self) -> int:

        """Dimensionality of the input data."""

        return self._dim

    ####################################################################################################################

    @property
    def dtype(self) -> typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]:

        """Neural network data type."""

        return self._dtype

    ####################################################################################################################

    @property
    def topology(self) -> str:

        """Neural network topology, either **'square'** or **'hexagonal'**."""

        return self._topology

    ####################################################################################################################

    @property
    def weights(self) -> np.ndarray:

        """Neural network weights with the shape `[m * n, dim]`."""

        return self._weights.reshape((self._m * self._n, self._dim))

    ####################################################################################################################

    @property
    def centroids(self) -> np.ndarray:

        """Neural network weights with the shape `[m, n, dim]`."""

        return self._weights.reshape((self._m, self._n, self._dim))

    ####################################################################################################################

    @property
    def quantization_errors(self) -> np.ndarray:

        """
        Returns the quantization error.

        .. math::
            c_i^1\\equiv\\mathrm{1^\\mathrm{st}\\,bmu}\\equiv\\underset{j}{\\mathrm{arg\\,min}_1}\\lVert x_i-w_j\\rVert

        .. math::
            \\boxed{e_Q\\equiv\\frac{1}{N}\\sum_{i=1}^N\\lVert x_i-w_{c_i^1}\\rVert}

        """

        return self._quantization_errors

    ####################################################################################################################

    @property
    def topographic_errors(self) -> np.ndarray:

        """
        Returns the topographic errors.

        .. math::
            c_i^n\\equiv\\mathrm{n^\\mathrm{th}\\,bmu}\\equiv\\underset{j}{\\mathrm{arg\\,min}_n}\\lVert x_i-w_j\\rVert

        .. math::
            r\\equiv\\left\\{\\begin{array}{ll}\\sqrt{1}&\\mathrm{topology=hexagon}\\\\\\sqrt{2}&\\mathrm{topology=square}\\end{array}\\right.

        .. math::
            t(x_i)\\equiv\\left\\{\\begin{array}{ll}1&\\lVert c_i^1-c_i^2\\rVert>r\\\\0&\\mathrm{otherwise}\\end{array}\\right.

        .. math::
            \\boxed{e_t\\equiv\\frac{1}{N}\\sum_{i=0}^Nt(x_i)}
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
    @nb.njit()
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

    def get_distance_map(self, normalization: typing.Optional[str] = None) -> np.ndarray:

        """
        Returns the distance map of the weights.

        Parameters
        ----------
        normalization : str, default: **None** ≡ '**sum**'
            Normalization method, either '**sum**' or '**mean**'.
        """

        normalization = normalization or 'sum'

        ################################################################################################################

        if self._topology == 'square':

            result = np.full((self._m, self._n, 8), np.nan, dtype = self._dtype)

            SOM_Abstract._distance_map(result, self.centroids, SOM_Abstract._X_SQU_STENCIL, SOM_Abstract._Y_SQU_STENCIL, self._m, self._n, 8)

        else:

            result = np.full((self._m, self._n, 6), np.nan, dtype = self._dtype)

            SOM_Abstract._distance_map(result, self.centroids, SOM_Abstract._X_HEX_STENCIL, SOM_Abstract._Y_HEX_STENCIL, self._m, self._n, 6)

        ################################################################################################################

        if normalization == 'sum':
            result = np.nansum(result, axis = 2)

        elif normalization == 'mean':
            result = np.nanmean(result, axis = 2)

        else:
            raise ValueError(f'Invalid normalization method `{normalization}`')

        ################################################################################################################

        return result / np.max(result)

    ####################################################################################################################

    def compute_errors(self, dataset: typing.Union[np.ndarray, typing.Callable], show_progress_bar: bool = False, enable_gpu: bool = True, threads_per_blocks: typing.Optional[int] = None) -> typing.Tuple[float, float]:

        """
        For the given input, computes the quantization and topographic errors.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Dataset array or generator builder.
        show_progress_bar : bool, default: **False**
            Specifies whether to display a progress bar.
        enable_gpu : bool, default: **True**
            If available, run on GPU rather than CPU.
        threads_per_blocks : int, default: **1024**
            Number of GPU threads per blocks.
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        result = device_array_empty(shape = 2, dtype = np.float32)

        ################################################################################################################

        n_vectors = 0

        generator = generator_builder()

        for vectors in tqdm.tqdm(generator(), disable = not show_progress_bar):

            n_vectors += vectors.shape[0]

            _compute_errors_kernel[enable_gpu, threads_per_blocks, vectors.shape[0]](result, self._weights, self._topography, vectors, 2.0 if self._topology == 'square' else 1.0, self._m * self._n)

        ################################################################################################################

        return result.copy_to_host() / n_vectors

    ####################################################################################################################

    def get_activation_map(self, dataset: typing.Union[np.ndarray, typing.Callable], dataset_weights: typing.Optional[typing.Union[np.ndarray, typing.Callable]] = None, show_progress_bar: bool = False, enable_gpu: bool = True, threads_per_blocks: typing.Optional[int] = None) -> np.ndarray:

        """
        For the given input, returns a matrix where the element i,j is the number of times that the neuron i,j have been activated.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Dataset array or generator builder.
        dataset_weights : typing.Union[np.ndarray, typing.Callable], default: **None**
            Training dataset weight array or generator builder.
        show_progress_bar : bool, default: **False**
            Specifies whether to display a progress bar.
        enable_gpu : bool, default: **True**
            If available, run on GPU rather than CPU.
        threads_per_blocks : int, default: **1024**
            Number of GPU threads per blocks.
        """

        ################################################################################################################

        dataset_generator_builder = dataset_to_generator_builder(    dataset    )
        density_generator_builder = dataset_to_generator_builder(dataset_weights)

        ################################################################################################################

        result = device_array_zeros(shape = (self._m * self._n, ), dtype = np.int64)

        ################################################################################################################

        if density_generator_builder is not None:

            dataset_generator = dataset_generator_builder()
            density_generator = density_generator_builder()

            for vectors, density in tqdm.tqdm(zip(dataset_generator(), density_generator()), disable = not show_progress_bar):

                _count_bmus_kernel[enable_gpu, threads_per_blocks, vectors.shape[0]](
                    result,
                    self._weights,
                    vectors,
                    density.astype(np.int64),
                    self._m * self._n
                )

        else:

            dataset_generator = dataset_generator_builder()

            for vectors in tqdm.tqdm(dataset_generator(), disable = not show_progress_bar):

                _count_bmus_kernel[enable_gpu, threads_per_blocks, vectors.shape[0]](
                    result,
                    self._weights,
                    vectors,
                    np.ones(vectors.shape[0], dtype = np.int64),
                    self._m * self._n
                )

        ################################################################################################################

        return result.copy_to_host().reshape((self._m, self._n))

    ####################################################################################################################

    def get_winners(self, dataset: np.ndarray, enable_gpu: bool = True, threads_per_blocks: typing.Optional[int] = None) -> np.ndarray:

        """
        For the given input, returns a vector of the best matching unit indices :math:`\\in[0,m\\times n-1]`.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset array.
        enable_gpu : bool, default: **True**
            If available, run on GPU rather than CPU.
        threads_per_blocks : int, default: **1024**
            Number of GPU threads per blocks.
        """

        ################################################################################################################

        result = device_array_empty(dataset.shape[0], dtype = np.int32)

        ################################################################################################################

        _find_bmus_kernel[enable_gpu, threads_per_blocks, dataset.shape[0]](result, self._weights, dataset, self._m * self._n)

        ################################################################################################################

        return result.copy_to_host()

########################################################################################################################

# noinspection DuplicatedCode
@jit(kernel = True, parallel = True)
def _count_bmus_kernel(result: np.ndarray, weights: np.ndarray, vectors: np.ndarray, density: np.ndarray, mn: int) -> None:

    if jit.is_gpu:

        ################################################################################################################
        # GPU                                                                                                          #
        ################################################################################################################

        i = jit.grid(1)
        if i < vectors.shape[0]:

            jit.atomic_add(result, _find_bmu_xpu(weights, vectors[i], mn), density[i])

        ################################################################################################################

    else:

        ################################################################################################################
        # CPU                                                                                                          #
        ################################################################################################################

        for i in nb.prange(vectors.shape[0]):

            jit.atomic_add(result, _find_bmu_xpu(weights, vectors[i], mn), density[i])

########################################################################################################################

@jit(kernel = True, parallel = True)
def _find_bmus_kernel(result: np.ndarray, weights: np.ndarray, vectors: np.ndarray, mn: int) -> None:

    if jit.is_gpu:

        ################################################################################################################
        # GPU                                                                                                          #
        ################################################################################################################

        i = jit.grid(1)
        if i < vectors.shape[0]:

            result[i] = _find_bmu_xpu(weights, vectors[i], mn)

        ################################################################################################################

    else:

        ################################################################################################################
        # CPU                                                                                                          #
        ################################################################################################################

        for i in nb.prange(vectors.shape[0]):

            result[i] = _find_bmu_xpu(weights, vectors[i], mn)

########################################################################################################################

@jit(fastmath = True)
def _find_bmu_xpu(weights: np.ndarray, vector: np.ndarray, mn: int) -> int:

    min_distance = 1.0e99
    min_index = 0x00

    for index in range(mn):

        distance = square_distance_xpu(weights[index], vector)

        if min_distance > distance:

            min_distance = distance
            min_index = index

    return min_index

########################################################################################################################

@jit(kernel = True, parallel = True)
def _compute_errors_kernel(errors: np.ndarray, weights: np.ndarray, topography: np.ndarray, vectors: np.ndarray, penalty_dist: float, mn: int) -> None:

    if jit.is_gpu:

        ################################################################################################################
        # GPU                                                                                                          #
        ################################################################################################################

        i = jit.grid(1)
        if i < vectors.shape[0]:

            _compute_errors_xpu(errors, weights, topography, vectors[i], penalty_dist, mn)

        ################################################################################################################

    else:

        ################################################################################################################
        # CPU                                                                                                          #
        ################################################################################################################

        for i in nb.prange(vectors.shape[0]):

            _compute_errors_xpu(errors, weights, topography, vectors[i], penalty_dist, mn)

########################################################################################################################

@jit(fastmath = True)
def _compute_errors_xpu(errors: np.ndarray, weights: np.ndarray, topography: np.ndarray, vector: np.ndarray, penalty_dist: float, mn: int) -> None:

    ####################################################################################################################
    # DO BMUS CALCULATION                                                                                              #
    ####################################################################################################################

    ###_distance2 = 1.0e99
    min_distance1 = 1.0e99

    min_index2 = 0
    min_index1 = 0

    for min_index0 in range(mn):

        min_distance0 = square_distance_xpu(weights[min_index0], vector)

        if min_distance1 > min_distance0:

            ###_distance2 = min_distance1
            min_distance1 = min_distance0

            min_index2 = min_index1
            min_index1 = min_index0

    ####################################################################################################################

    bmu2 = topography[min_index2]
    bmu1 = topography[min_index1]

    ####################################################################################################################
    # UPDATE ERRORS                                                                                                    #
    ####################################################################################################################

    if square_distance_xpu(bmu1, bmu2) > penalty_dist:

        jit.atomic_add(errors, 1, 1.0000000000000000000000)

    jit.atomic_add(errors, 0, math.sqrt(min_distance1))

########################################################################################################################
