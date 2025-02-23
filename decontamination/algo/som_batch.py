# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import gc
import math
import tqdm
import typing

import numpy as np
import numba as nb

from .. import jit, device_array_zeros

from . import som_abstract, square_distance_xpu, asymptotic_decay_cpu, asymptotic_decay_gpu, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class SOM_Batch(som_abstract.SOM_Abstract):

    """
    Self Organizing Maps (standard batch implementation). It runs with constant memory usage.

    .. note::
        A rule of thumb to set the size of the grid for a dimensionality reduction task is that it should contain :math:`5\\sqrt{N}` neurons where N is the number of samples in the dataset to analyze.

    Parameters
    ----------
    m : int
        Number of neuron rows.
    n : int
        Number of neuron columns.
    dim : int
        Dimensionality of the input data.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]], default:  **np.float32**
        Neural network data type, either **np.float32** or **np.float64**.
    topology : str, default: **None** ≡ **'hexagonal'**
        Neural network topology, either **'square'** or **'hexagonal'**.
    sigma : float, default: default: **None** ≡ :math:`\\mathrm{max}(m,n)/2`
        Starting value of the neighborhood radius.
    """

    __MODE__ = 'batch'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, topology: typing.Optional[str] = None, sigma: float = None):

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology)

        ################################################################################################################

        self._sigma = max(m, n) / 2.0 if sigma is None else float(sigma)

        ################################################################################################################

        self._n_epochs = None

        self._n_vectors = None

        ################################################################################################################

        self._header_extra = {
            'mode': '__MODE__',
            'sigma': '_sigma',
            'n_epochs': '_n_epochs',
            'n_vectors': '_n_vectors',
        }

    ####################################################################################################################

    @property
    def sigma(self) -> float:

        """Starting value of the neighborhood radius."""

        return self._sigma

    ####################################################################################################################

    @staticmethod
    @jit(kernel = True, parallel = True)
    def _train_step1_epoch_kernel(numerator: np.ndarray, denominator: np.ndarray, weights: np.ndarray, topography: np.ndarray, vectors: np.ndarray, density: np.ndarray, cur_epoch: int, n_epochs: int, sigma0: float, mn: int) -> None:

        if jit.is_gpu:

            ############################################################################################################
            # GPU                                                                                                      #
            ############################################################################################################

            i = jit.grid(1)
            if i < vectors.shape[0]:

                sigma = sigma0 * asymptotic_decay_gpu(cur_epoch, n_epochs)

                _train_step2_xpu(
                    numerator,
                    denominator,
                    weights,
                    topography,
                    vectors[i],
                    density[i],
                    sigma,
                    mn
                )

            ############################################################################################################

        else:

            ############################################################################################################
            # CPU                                                                                                      #
            ############################################################################################################

            sigma = sigma0 * asymptotic_decay_cpu(cur_epoch, n_epochs)

            for i in nb.prange(vectors.shape[0]):

                _train_step2_xpu(
                    numerator,
                    denominator,
                    weights,
                    topography,
                    vectors[i],
                    density[i],
                    sigma,
                    mn
                )

            ############################################################################################################

        jit.syncthreads()

    ####################################################################################################################

    @staticmethod
    @jit(kernel = True, parallel = True)
    def _train_step1_iter_kernel(numerator: np.ndarray, denominator: np.ndarray, weights: np.ndarray, topography: np.ndarray, vectors: np.ndarray, density: np.ndarray, cur_vector: int, n_vectors: int, sigma0: float, mn: int) -> None:

        if jit.is_gpu:

            ############################################################################################################
            # GPU                                                                                                      #
            ############################################################################################################

            i = jit.grid(1)
            if i < vectors.shape[0]:

                sigma = sigma0 * asymptotic_decay_gpu(cur_vector + i, n_vectors)

                _train_step2_xpu(
                    numerator,
                    denominator,
                    weights,
                    topography,
                    vectors[i],
                    density[i],
                    sigma,
                    mn
                )

            ############################################################################################################

        else:

            ############################################################################################################
            # CPU                                                                                                      #
            ############################################################################################################

            for i in nb.prange(vectors.shape[0]):

                sigma = sigma0 * asymptotic_decay_cpu(cur_vector + i, n_vectors)

                _train_step2_xpu(
                    numerator,
                    denominator,
                    weights,
                    topography,
                    vectors[i],
                    density[i],
                    sigma,
                    mn
                )

            ############################################################################################################

        jit.syncthreads()

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], dataset_weights: typing.Optional[typing.Union[np.ndarray, typing.Callable]] = None, n_epochs: typing.Optional[int] = None, n_vectors: typing.Optional[int] = None, use_best_epoch: bool = True, stop_quantization_error: typing.Optional[float] = None, stop_topographic_error: typing.Optional[float] = None, show_progress_bar: bool = False, enable_gpu: bool = True, threads_per_blocks: typing.Optional[int] = None) -> None:

        """
        Trains the neural network. Use either the "*number of epochs*" training method by specifying `n_epochs` (then :math:`e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_epochs}\\}-1`) or the "*number of vectors*" training method by specifying `n_vectors` (then :math:`e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_vectors}\\}-1`). A batch formulation of updating weights is implemented:

        .. math::
            c_i(x,w,e)\\equiv\\mathrm{bmu}(x_i,w(e))\\equiv\\underset{j}{\\mathrm{arg\\,min}}\\lVert x_i-w_j(e)\\rVert

        .. math::
            \\Theta_{ji}(x,w,e)\\equiv\\exp\\left(-\\frac{\\lVert j-c_i(x,w,e)\\rVert^2}{2\\sigma^2(e)}\\right)

        .. math::
            \\boxed{w_j(e+1)=\\frac{\\sum_{i=0}^{N-1}\\Theta_{ji}(x,w,e)\\cdot x_i}{\\sum_{i=0}^{N-1}\\Theta_{ji}(x,w,e)}}

        where :math:`j=0\\dots m\\times n-1`, at epoch :math:`e`, :math:`\\sigma(e)\\equiv\\sigma\\cdot\\frac{1}{1+2\\frac{e}{e_\\mathrm{tot}}}` is the neighborhood radius.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        dataset_weights : typing.Union[np.ndarray, typing.Callable], default: **None**
            Training dataset weight array or generator builder.
        n_epochs : int, default: **None**
            Optional number of epochs to train for.
        n_vectors : int, default: **None**
            Optional number of vectors to train for.
        use_best_epoch : bool, default: **True**
            ???
        stop_quantization_error : float, default: **None**
            Stops training if quantization_error < stop_quantization_error.
        stop_topographic_error : float, default: **None**
            Stops training if topographic_error < stop_topographic_error.
        show_progress_bar : bool, default: **False**
            Specifies whether to display a progress bar.
        enable_gpu : bool, default: **True**
            If available, run on GPU rather than CPU.
        threads_per_blocks : int, default: **None** ≡ maximum
            Number of GPU threads per blocks.
        """

        if stop_quantization_error is None:
            stop_quantization_error = -1.0e6

        if stop_topographic_error is None:
            stop_topographic_error = -1.0e6

        ################################################################################################################

        dataset_generator_builder = dataset_to_generator_builder(    dataset    )
        density_generator_builder = dataset_to_generator_builder(dataset_weights)

        ################################################################################################################

        self._n_epochs = n_epochs

        self._n_vectors = n_vectors

        if not (n_epochs is None) and (n_vectors is None):

            ############################################################################################################
            # TRAINING BY NUMBER OF EPOCHS                                                                             #
            ############################################################################################################

            self._history = np.empty((n_epochs, self._m * self._n, self._dim), dtype = self._dtype)

            self._quantization_errors = np.full(n_epochs, np.nan, dtype = np.float64)

            self._topographic_errors = np.full(n_epochs, np.nan, dtype = np.float64)

            ############################################################################################################

            for cur_epoch in tqdm.trange(n_epochs, disable = not show_progress_bar):

                ########################################################################################################

                numerator = device_array_zeros(shape = (self._m * self._n, self._dim), dtype = self._dtype)

                denominator = device_array_zeros(shape = (self._m * self._n, ), dtype = self._dtype)

                ########################################################################################################

                if density_generator_builder is not None:

                    dataset_generator = dataset_generator_builder()
                    density_generator = density_generator_builder()

                    for vectors, density in zip(dataset_generator(), density_generator()):

                        SOM_Batch._train_step1_epoch_kernel[enable_gpu, threads_per_blocks, vectors.shape[0]](
                            numerator,
                            denominator,
                            self._weights,
                            self._topography,
                            vectors.astype(self._dtype),
                            density.astype(np.int32),
                            cur_epoch,
                            n_epochs,
                            self._dtype(self._sigma),
                            self._m * self._n
                        )

                        gc.collect()

                else:

                    dataset_generator = dataset_generator_builder()

                    for vectors in dataset_generator():

                        SOM_Batch._train_step1_epoch_kernel[enable_gpu, threads_per_blocks, vectors.shape[0]](
                            numerator,
                            denominator,
                            self._weights,
                            self._topography,
                            vectors.astype(self._dtype),
                            np.ones(vectors.shape[0], dtype = np.int32),
                            cur_epoch,
                            n_epochs,
                            self._dtype(self._sigma),
                            self._m * self._n
                        )

                        gc.collect()

                ########################################################################################################

                numerator_host = numerator.copy_to_host()

                denominator_temp = denominator.copy_to_host()

                denominator_host = np.expand_dims(denominator_temp, axis = -1)

                ########################################################################################################

                self._weights = np.divide(
                    numerator_host,
                    denominator_host,
                    out = np.zeros_like(numerator_host),
                    where = denominator_host != 0.0
                )

                ############################################################################################################

                self._history[cur_epoch] = self._weights

                ########################################################################################################

                errors = self.compute_errors(dataset, show_progress_bar = False, enable_gpu = enable_gpu, threads_per_blocks = threads_per_blocks)

                self._quantization_errors[cur_epoch] = errors[0]
                self._topographic_errors[cur_epoch] = errors[1]

                if errors[0] <= stop_quantization_error\
                   and                                 \
                   errors[1] <= stop_topographic_error:

                    print('Stopping at epoch #{}.'.format(cur_epoch))

                    break

            ############################################################################################################

        elif (n_epochs is None) and not (n_vectors is None):

            cur_vector = 0

            ############################################################################################################
            # TRAINING BY NUMBER OF VECTORS                                                                            #
            ############################################################################################################

            self._history = np.empty((1, self._m * self._n, self._dim), dtype = self._dtype)

            self._quantization_errors = np.zeros(1, dtype = np.float64)

            self._topographic_errors = np.zeros(1, dtype = np.float64)

            ############################################################################################################

            numerator = device_array_zeros(shape = (self._m * self._n, self._dim), dtype = self._dtype)

            denominator = device_array_zeros(shape = (self._m * self._n, ), dtype = self._dtype)

            ############################################################################################################

            progress_bar = tqdm.tqdm(total = n_vectors, disable = not show_progress_bar)

            if density_generator_builder is not None:

                dataset_generator = dataset_generator_builder()
                density_generator = density_generator_builder()

                for vectors, density in zip(dataset_generator(), density_generator()):

                    count = min(vectors.shape[0], n_vectors - cur_vector)

                    SOM_Batch._train_step1_iter_kernel[enable_gpu, threads_per_blocks, count](
                        numerator,
                        denominator,
                        self._weights,
                        self._topography,
                        vectors[0: count].astype(self._dtype),
                        density[0: count].astype(np.int32),
                        cur_vector,
                        n_vectors,
                        self._dtype(self._sigma),
                        self._m * self._n
                    )

                    gc.collect()

                    cur_vector += count

                    progress_bar.update(count)

                    if cur_vector >= n_vectors:

                        break

            else:

                dataset_generator = dataset_generator_builder()

                for vectors in dataset_generator():

                    count = min(vectors.shape[0], n_vectors - cur_vector)

                    SOM_Batch._train_step1_iter_kernel[enable_gpu, threads_per_blocks, count](
                        numerator,
                        denominator,
                        self._weights,
                        self._topography,
                        vectors[0: count].astype(self._dtype),
                        np.ones(count, dtype = np.int32),
                        cur_vector,
                        n_vectors,
                        self._dtype(self._sigma),
                        self._m * self._n
                    )

                    gc.collect()

                    cur_vector += count

                    progress_bar.update(count)

                    if cur_vector >= n_vectors:

                        break

            ############################################################################################################

            numerator_host = numerator.copy_to_host()

            denominator_temp = denominator.copy_to_host()

            denominator_host = np.expand_dims(denominator_temp, axis = -1)

            ############################################################################################################

            self._weights = np.divide(
                numerator_host,
                denominator_host,
                out = np.zeros_like(numerator_host),
                where = denominator_host != 0.0
            )

            ############################################################################################################

            self._history[0] = self._weights

            ############################################################################################################

            errors = self.compute_errors(dataset, show_progress_bar = False, enable_gpu = enable_gpu, threads_per_blocks = threads_per_blocks)

            self._quantization_errors[0] = errors[0]
            self._topographic_errors[0] = errors[1]

            ############################################################################################################

        else:

            raise ValueError('Invalid training method, specify either `n_epochs` or `n_vectors`')

        ################################################################################################################

        if use_best_epoch:

            self._weights = self._history[np.argmin(self._quantization_errors)]

########################################################################################################################

@jit(fastmath = True)
def _train_step2_xpu(numerator: np.ndarray, denominator: np.ndarray, weights: np.ndarray, topography: np.ndarray, vector: np.ndarray, density: np.ndarray, sigma: float, mn: int) -> None:

    ####################################################################################################################
    # DO BMUS CALCULATION                                                                                              #
    ####################################################################################################################

    min_distance = 1.0e99
    min_index = 0

    for index in range(mn):

        distance = square_distance_xpu(weights[index], vector)

        if min_distance > distance:

            min_distance = distance
            min_index = index

    ####################################################################################################################

    bmu = topography[min_index]

    ####################################################################################################################
    # UPDATE WEIGHTS                                                                                                   #
    ####################################################################################################################

    for i in range(mn):

        ################################################################################################################

        neighborhood_i = math.exp(-square_distance_xpu(topography[i], bmu) / (2.0 * sigma ** 2))

        ################################################################################################################

        numerator_i = numerator[i]

        for k in range(vector.shape[0]):

            jit.atomic_add(numerator_i, k, density * neighborhood_i * vector[k])

        jit.atomic_add(denominator, i, density * neighborhood_i * 1.0000000)

########################################################################################################################
