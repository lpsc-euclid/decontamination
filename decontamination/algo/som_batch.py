# -*- coding: utf-8 -*-
########################################################################################################################

import gc
import math
import tqdm
import typing

import numpy as np
import numba as nb

from .. import jit, device_array_empty, device_array_zeros

from . import som_abstract, square_distance_xpu, atomic_add_vector_xpu, asymptotic_decay_cpu, asymptotic_decay_gpu, dataset_to_generator_builder

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
    topology : typing.Optional[str], default: **None** ≡ **'hexagonal'**
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
    def _train_step1_epoch_kernel(numerator: np.ndarray, denominator: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, weights: np.ndarray, topography: np.ndarray, vectors: np.ndarray, cur_epoch: int, n_epochs: int, sigma0: float, penalty_dist: float, mn: int) -> None:

        ################################################################################################################
        # !--BEGIN-CPU--

        sigma = sigma0 * asymptotic_decay_cpu(cur_epoch, n_epochs)

        for i in nb.prange(vectors.shape[0]):

            _train_step2_xpu(
                numerator,
                denominator,
                quantization_errors,
                topographic_errors,
                weights,
                topography,
                vectors[i],
                sigma,
                penalty_dist,
                cur_epoch,
                mn
            )

        # !--END-CPU--
        ####################################################################################################################
        # !--BEGIN-GPU--

        i = jit.grid(1)

        if i < vectors.shape[0]:

            sigma = sigma0 * asymptotic_decay_gpu(cur_epoch, n_epochs)

            _train_step2_xpu(
                numerator,
                denominator,
                quantization_errors,
                topographic_errors,
                weights,
                topography,
                vectors[i],
                sigma,
                penalty_dist,
                cur_epoch,
                mn
            )

        # !--END-GPU--
        ################################################################################################################

        jit.syncthreads()

    ####################################################################################################################

    @staticmethod
    @jit(kernel = True, parallel = True)
    def _train_step1_iter_kernel(numerator: np.ndarray, denominator: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, weights: np.ndarray, topography: np.ndarray, vectors: np.ndarray, cur_vector: int, n_vectors: int, n_err_bins: int, sigma0: float, penalty_dist: float, mn: int) -> None:

        ################################################################################################################
        # !--BEGIN-CPU--

        for i in nb.prange(vectors.shape[0]):

            cur_err_bin = (n_err_bins * (cur_vector + i)) // n_vectors

            sigma = sigma0 * asymptotic_decay_cpu(cur_vector + i, n_vectors)

            _train_step2_xpu(
                numerator,
                denominator,
                quantization_errors,
                topographic_errors,
                weights,
                topography,
                vectors[i],
                sigma,
                penalty_dist,
                cur_err_bin,
                mn
            )

        # !--END-CPU--
        ####################################################################################################################
        # !--BEGIN-GPU--

        i = jit.grid(1)

        if i < vectors.shape[0]:

            cur_err_bin = (n_err_bins * (cur_vector + i)) // n_vectors

            sigma = sigma0 * asymptotic_decay_gpu(cur_vector + i, n_vectors)

            _train_step2_xpu(
                numerator,
                denominator,
                quantization_errors,
                topographic_errors,
                weights,
                topography,
                vectors[i],
                sigma,
                penalty_dist,
                cur_err_bin,
                mn
            )

        # !--END-GPU--
        ################################################################################################################

        jit.syncthreads()

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], n_epochs: typing.Optional[int] = None, n_vectors: typing.Optional[int] = None, n_error_bins: int = 10, show_progress_bar: bool = False, enable_gpu: bool = True, threads_per_blocks: int = 1024) -> None:

        """
        Trains the neural network. Use either the "*number of epochs*" training method by specifying `n_epochs` (then :math:`e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_epochs}\\}-1`) or the "*number of vectors*" training method by specifying `n_vectors` (then :math:`e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_vectors}\\}-1`). A batch formulation of updating weights is implemented:

        .. math::
            c_i(w,e)\\equiv\\mathrm{bmu}(x_i,w,e)\\equiv\\underset{j}{\\mathrm{arg\\,min}}\\lVert x_i-w_j(e)\\rVert

        if :math:`\\sigma>0`:

        .. math::
            \\Theta_{ji}(w,e)\\equiv\\exp\\left(-\\frac{\\lVert j-c_i(w,e)\\rVert^2}{2\\sigma^2(e)}\\right)

        if :math:`\\sigma=0`:

        .. math::
            \\Theta_{ji}(w,e)\\equiv\\delta_{j,c_i(w,e)}\\equiv\\left\\{\\begin{array}{ll}1&j=c_i(w,e)\\\\0&\\mathrm{otherwise}\\end{array}\\right.

        .. math::
            \\boxed{w_j(e+1)=\\frac{\\sum_{i=0}^{N-1}\\Theta_{ji}(w,e)\\cdot x_i}{\\sum_{i=0}^{N-1}\\Theta_{ji}(w,e)}}

        where :math:`j=0\\dots m\\times n-1`, at epoch :math:`e`, :math:`\\sigma(e)\\equiv\\sigma\\cdot\\frac{1}{1+2\\frac{e}{e_\\mathrm{tot}}}` is the neighborhood radius.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        n_epochs : typing.Optional[int], default: **None**
            Number of epochs to train for.
        n_vectors : typing.Optional[int], default: **None**
            Number of vectors to train for.
        n_error_bins : int, default: **10**
            Number of quantization and topographic error bins.
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

        cur_vector = 0

        self._n_epochs = n_epochs

        self._n_vectors = n_vectors

        penalty_dist = 2.0 if self._topology == 'square' else 1.0

        if not (n_epochs is None) and (n_vectors is None):

            ############################################################################################################
            # TRAINING BY NUMBER OF EPOCHS                                                                             #
            ############################################################################################################

            quantization_errors = device_array_empty(n_epochs, dtype = np.float32)

            topographic_errors = device_array_empty(n_epochs, dtype = np.float32)

            ############################################################################################################

            for cur_epoch in tqdm.trange(n_epochs, disable = not show_progress_bar):

                ########################################################################################################

                numerator = device_array_zeros(shape = (self._m * self._n, self._dim), dtype = self._dtype)

                denominator = device_array_zeros(shape = (self._m * self._n, ), dtype = self._dtype)

                ########################################################################################################

                generator = generator_builder()

                for vectors in generator():

                    cur_vector += vectors.shape[0]

                    SOM_Batch._train_step1_epoch_kernel[enable_gpu, threads_per_blocks, vectors.shape[0]](
                        numerator,
                        denominator,
                        quantization_errors,
                        topographic_errors,
                        self._weights,
                        self._topography,
                        vectors,
                        cur_epoch,
                        n_epochs,
                        self._sigma,
                        penalty_dist,
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

            if cur_vector > 0:

                self._quantization_errors = quantization_errors.copy_to_host() * n_epochs / cur_vector

                self._topographic_errors = topographic_errors.copy_to_host() * n_epochs / cur_vector

            ############################################################################################################

        elif (n_epochs is None) and not (n_vectors is None):

            ############################################################################################################
            # TRAINING BY NUMBER OF VECTORS                                                                            #
            ############################################################################################################

            quantization_errors = device_array_empty(n_error_bins, dtype = np.float32)

            topographic_errors = device_array_empty(n_error_bins, dtype = np.float32)

            ############################################################################################################

            numerator = device_array_zeros(shape = (self._m * self._n, self._dim), dtype = self._dtype)

            denominator = device_array_zeros(shape = (self._m * self._n, ), dtype = self._dtype)

            ########################################################################################################

            progress_bar = tqdm.tqdm(total = n_vectors, disable = not show_progress_bar)

            generator = generator_builder()

            for vectors in generator():

                count = min(vectors.shape[0], n_vectors - cur_vector)

                SOM_Batch._train_step1_iter_kernel[enable_gpu, threads_per_blocks, vectors.shape[0]](
                    numerator,
                    denominator,
                    quantization_errors,
                    topographic_errors,
                    self._weights,
                    self._topography,
                    vectors[0: count],
                    cur_vector,
                    n_vectors,
                    n_error_bins,
                    self._sigma,
                    penalty_dist,
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

            if cur_vector > 0:

                self._quantization_errors = quantization_errors.copy_to_host() * n_error_bins / cur_vector

                self._topographic_errors = topographic_errors.copy_to_host() * n_error_bins / cur_vector

            ############################################################################################################

        else:

            raise ValueError('Invalid training method, specify either `n_epochs` or `n_vectors`.')

########################################################################################################################

@jit(fastmath = True)
def _train_step2_xpu(numerator: np.ndarray, denominator: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, weights: np.ndarray, topography: np.ndarray, vector: np.ndarray, sigma: float, penalty_dist: float, err_bin: int, mn: int) -> None:

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
    # UPDATE WEIGHTS                                                                                                   #
    ####################################################################################################################

    if sigma > 0.0:

        ################################################################################################################
        # ... WITH GAUSSIAN NEIGHBORHOOD OPERATOR                                                                      #
        ################################################################################################################

        for i in range(mn):

            ############################################################################################################

            neighborhood_i = math.exp(-square_distance_xpu(topography[i], bmu1) / (2.0 * sigma ** 2))

            ############################################################################################################

            numerator_i = numerator[i]

            for k in range(vector.shape[0]):

                jit.atomic_add(numerator_i, k, neighborhood_i * vector[k])

            jit.atomic_add(denominator, i, neighborhood_i * 1.0000000)

        ################################################################################################################

    else:

        ################################################################################################################
        # ... WITH DIRAC NEIGHBORHOOD OPERATOR                                                                         #
        ################################################################################################################

        atomic_add_vector_xpu(numerator[min_index1], vector)

        jit.atomic_add(denominator, min_index1, 1.0000)

    ####################################################################################################################
    # UPDATE ERRORS                                                                                                    #
    ####################################################################################################################

    if square_distance_xpu(bmu1, bmu2) > penalty_dist:

        jit.atomic_add(topographic_errors, err_bin, 1.0000000000000000000000)

    jit.atomic_add(quantization_errors, err_bin, math.sqrt(min_distance1))

########################################################################################################################
