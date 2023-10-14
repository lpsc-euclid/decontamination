# -*- coding: utf-8 -*-
########################################################################################################################

import math
import tqdm
import typing

import numpy as np
import numba as nb

from . import som_abstract, asymptotic_decay_cpu, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class SOM_Online(som_abstract.SOM_Abstract):

    """
    Self Organizing Maps (standard online implementation).

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
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]
        Neural network data type (default: **np.float32**).
    topology : typing.Optional[str]
        Topology of the model, either **'square'** or **'hexagonal'** (default: **None**, uses: **'hexagonal'**).
    alpha : float
        Starting value of the learning rate (default: **None**, uses: **0.3**).
    sigma : float
        Starting value of the neighborhood radius (default: **None**, uses: :math:`\\mathrm{max}(m,n)/2`).
    """

    __MODE__ = 'online'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, topology: typing.Optional[str] = None, alpha: float = None, sigma: float = None):

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology)

        ################################################################################################################

        self._alpha = 0.3 if alpha is None else float(alpha)

        self._sigma = max(m, n) / 2.0 if sigma is None else float(sigma)

        ################################################################################################################

        self._n_epochs = None

        self._n_vectors = None

        ################################################################################################################

        self._header_extra = {
            'mode': '__MODE__',
            'alpha': '_alpha',
            'sigma': '_sigma',
            'n_epochs': '_n_epochs',
            'n_vectors': '_n_vectors',
        }

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _train_step1_epoch(weights: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, topography: np.ndarray, vectors: np.ndarray, cur_epoch: int, n_epochs: int, alpha0: float, sigma0: float, penalty_dist: float, mn: int):

        ################################################################################################################

        decay_function = asymptotic_decay_cpu(cur_epoch, n_epochs)

        alpha = alpha0 * decay_function

        sigma = sigma0 * decay_function

        ################################################################################################################

        for i in range(vectors.shape[0]):

            _train_step2(
                quantization_errors,
                topographic_errors,
                weights,
                topography,
                vectors[i],
                alpha,
                sigma,
                penalty_dist,
                cur_epoch,
                mn
            )

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _train_step1_iter(weights: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, topography: np.ndarray, vectors: np.ndarray, cur_vector: int, n_vectors: int, n_err_bins: int, alpha0: float, sigma0: float, penalty_dist: float, mn: int):

        for i in range(vectors.shape[0]):

            ############################################################################################################

            cur_err_bin = (n_err_bins * (cur_vector + i)) // n_vectors

            ############################################################################################################

            decay_function = asymptotic_decay_cpu(cur_vector + i, n_vectors)

            alpha = alpha0 * decay_function

            sigma = sigma0 * decay_function

            ############################################################################################################

            _train_step2(
                quantization_errors,
                topographic_errors,
                weights,
                topography,
                vectors[i],
                alpha,
                sigma,
                penalty_dist,
                cur_err_bin,
                mn
            )

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], n_epochs: typing.Optional[int] = None, n_vectors: typing.Optional[int] = None, n_error_bins: int = 10, show_progress_bar: bool = False) -> None:

        """
        Trains the neural network. Use either the "*number of epochs*" training method by specifying `n_epochs` (then :math:`e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_epochs}\\}-1`) or the "*number of vectors*" training method by specifying `n_vectors` (then :math:`e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_vectors}\\}-1`). An online formulation of updating weights is implemented:

        .. math::
            c_i(w,e)\\equiv\\mathrm{bmu}(x_i,w,e)\\equiv\\underset{j}{\\mathrm{arg\\,min}}\\lVert x_i-w_j(e)\\rVert

        .. math::
            \\Theta_{ji}(w,e)\\equiv\\exp\\left(-\\frac{\\lVert j-c_i(w,e)\\rVert^2}{2\\sigma^2(e)}\\right)

        .. math::
            \\boxed{\\mathrm{iteratively\\,for}\\,i=0\\dots N-1\\,\\mathrm{:}\\,w_j(e+1)=w_j(e)+\\alpha(e)\\cdot\\Theta_{ji}(w,e)[x_i-w_j(e)]}

        where :math:`j=0\\dots m\\times n-1` and, at epoch :math:`e`, :math:`\\alpha(e)\\equiv\\alpha\\cdot\\frac{1}{1+2\\frac{e}{e_\\mathrm{tot}}}` is the learning rate and :math:`\\sigma(e)\\equiv\\sigma\\cdot\\frac{1}{1+2\\frac{e}{e_\\mathrm{tot}}}` is the neighborhood radius.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        n_epochs : typing.Optional[int]
            Number of epochs to train for (default: **None**).
        n_vectors : typing.Optional[int]
            Number of vectors to train for (default: **None**).
        n_error_bins : int
            Number of quantization and topographic error bins (default: **10**).
        show_progress_bar : bool
            Specifies whether to display a progress bar (default: **False**).
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

            self._quantization_errors = np.zeros(n_epochs, dtype = np.float32)

            self._topographic_errors = np.zeros(n_epochs, dtype = np.float32)

            ############################################################################################################

            for cur_epoch in tqdm.trange(n_epochs, disable = not show_progress_bar):

                generator = generator_builder()

                for vectors in generator():

                    cur_vector += vectors.shape[0]

                    SOM_Online._train_step1_epoch(
                        self._weights,
                        self._quantization_errors,
                        self._topographic_errors,
                        self._topography,
                        vectors,
                        cur_epoch,
                        n_epochs,
                        self._alpha,
                        self._sigma,
                        penalty_dist,
                        self._m * self._n
                    )

            ############################################################################################################

            if cur_vector > 0:

                self._quantization_errors = self._quantization_errors * n_epochs / cur_vector

                self._topographic_errors = self._topographic_errors * n_epochs / cur_vector

            ############################################################################################################

        elif (n_epochs is None) and not (n_vectors is None):

            ############################################################################################################
            # TRAINING BY NUMBER OF VECTORS                                                                            #
            ############################################################################################################

            self._quantization_errors = np.zeros(n_error_bins, dtype = np.float32)

            self._topographic_errors = np.zeros(n_error_bins, dtype = np.float32)

            ############################################################################################################

            progress_bar = tqdm.tqdm(total = n_vectors, disable = not show_progress_bar)

            generator = generator_builder()

            for vectors in generator():

                count = min(vectors.shape[0], n_vectors - cur_vector)

                SOM_Online._train_step1_iter(
                    self._weights,
                    self._quantization_errors,
                    self._topographic_errors,
                    self._topography,
                    vectors[0: count],
                    cur_vector,
                    n_vectors,
                    n_error_bins,
                    self._alpha,
                    self._sigma,
                    penalty_dist,
                    self._m * self._n
                )

                cur_vector += count

                progress_bar.update(count)

                if cur_vector >= n_vectors:

                    break

            ############################################################################################################

            if cur_vector > 0:

                self._quantization_errors = self._quantization_errors * n_error_bins / cur_vector

                self._topographic_errors = self._topographic_errors * n_error_bins / cur_vector

            ############################################################################################################

        else:

            raise Exception('Invalid training method, specify either `n_epochs` or `n_vectors`.')

########################################################################################################################

@nb.njit(fastmath = True)
def _train_step2(quantization_errors: np.ndarray, topographic_errors: np.ndarray, weights: np.ndarray, topography: np.ndarray, vector: np.ndarray, alpha: float, sigma: float, penalty_dist: float, err_bin: int, mn: int) -> None:

    ####################################################################################################################
    # DO BMUS CALCULATION                                                                                              #
    ####################################################################################################################

    ###_distance2 = 1.0e99
    min_distance1 = 1.0e99

    min_index2 = 0
    min_index1 = 0

    for min_index0 in range(mn):

        min_distance0 = np.sum((weights[min_index0] - vector) ** 2)

        if min_distance1 > min_distance0:

            ###_distance2 = min_distance1
            min_distance1 = min_distance0

            min_index2 = min_index1
            min_index1 = min_index0

    ####################################################################################################################

    bmu2 = topography[min_index2]
    bmu1 = topography[min_index1]

    ####################################################################################################################
    # DO NEIGHBORHOOD OPERATOR CALCULATION                                                                             #
    ####################################################################################################################

    neighborhood_op = np.exp(-np.sum((topography - bmu1) ** 2, axis = -1) / (2.0 * sigma ** 2))

    ####################################################################################################################
    # UPDATE WEIGHTS                                                                                                   #
    ####################################################################################################################

    weights += alpha * np.expand_dims(neighborhood_op, -1) * (vector - weights)

    ####################################################################################################################
    # UPDATE ERRORS                                                                                                    #
    ####################################################################################################################

    if np.sum((bmu1 - bmu2) ** 2) > penalty_dist:

        topographic_errors[err_bin] += 1.0000000000000000000000

    quantization_errors[err_bin] += math.sqrt(min_distance1)

########################################################################################################################
