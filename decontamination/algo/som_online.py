# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import gc
import tqdm
import typing

import numpy as np
import numba as nb

from . import som_abstract, asymptotic_decay_cpu, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class SOM_Online(som_abstract.SOM_Abstract):

    """
    Self Organizing Maps (standard online implementation). It runs with constant memory usage.

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
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]], default: **np.float32**
        Neural network data type, either **np.float32** or **np.float64**.
    topology : str, default: **None** ≡ **'hexagonal'**
        Neural network topology, either **'square'** or **'hexagonal'**.
    alpha : float, default: **None** ≡ **0.3**
        Starting value of the learning rate.
    sigma : float, default: **None** ≡ :math:`\\mathrm{max}(m,n)/2`
        Starting value of the neighborhood radius.
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

    @property
    def alpha(self) -> float:

        """Starting value of the learning rate."""

        return self._alpha

    ####################################################################################################################

    @property
    def sigma(self) -> float:

        """Starting value of the neighborhood radius."""

        return self._sigma

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _train_step1_epoch(weights: np.ndarray, topography: np.ndarray, vectors: np.ndarray, cur_epoch: int, n_epochs: int, alpha0: float, sigma0: float, mn: int):

        ################################################################################################################

        decay_function = asymptotic_decay_cpu(cur_epoch, n_epochs)

        alpha = alpha0 * decay_function

        sigma = sigma0 * decay_function

        ################################################################################################################

        for i in range(vectors.shape[0]):

            _train_step2(
                weights,
                topography,
                vectors[i],
                alpha,
                sigma,
                mn
            )

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _train_step1_iter(weights: np.ndarray, topography: np.ndarray, vectors: np.ndarray, cur_vector: int, n_vectors: int, alpha0: float, sigma0: float, mn: int):

        for i in range(vectors.shape[0]):

            ############################################################################################################

            decay_function = asymptotic_decay_cpu(cur_vector + i, n_vectors)

            alpha = alpha0 * decay_function

            sigma = sigma0 * decay_function

            ############################################################################################################

            _train_step2(
                weights,
                topography,
                vectors[i],
                alpha,
                sigma,
                mn
            )

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], density: typing.Optional[typing.Union[np.ndarray, typing.Callable]] = None, n_epochs: typing.Optional[int] = None, n_vectors: typing.Optional[int] = None, stop_quantization_error: typing.Optional[float] = None, stop_topographic_error: typing.Optional[float] = None, show_progress_bar: bool = False, enable_gpu: bool = True, threads_per_blocks: int = 1024) -> None:

        """
        Trains the neural network. Use either the "*number of epochs*" training method by specifying `n_epochs` (then :math:`e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_epochs}\\}-1`) or the "*number of vectors*" training method by specifying `n_vectors` (then :math:`e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_vectors}\\}-1`). An online formulation of updating weights is implemented:

        .. math::
            c_i(x,w,t)\\equiv\\mathrm{bmu}(x_i,w(t))\\equiv\\underset{j}{\\mathrm{arg\\,min}}\\lVert x_i-w_j(t)\\rVert

        .. math::
            \\Theta_{ji}(x,w,e,t)\\equiv\\exp\\left(-\\frac{\\lVert j-c_i(x,w,t)\\rVert^2}{2\\sigma^2(e)}\\right)

        .. math::
            \\boxed{\\mathrm{iteratively\\,for}\\,t=0\\dots N-1\\,\\mathrm{:}\\,w_j(t+1)=w_j(t)+\\alpha(e)\\cdot\\Theta_{ji}(x,w,e,t)[x_i-w_j(t)]}

        where :math:`j=0\\dots m\\times n-1` and, at epoch :math:`e`, :math:`\\alpha(e)\\equiv\\alpha\\cdot\\frac{1}{1+2\\frac{e}{e_\\mathrm{tot}}}` is the learning rate and :math:`\\sigma(e)\\equiv\\sigma\\cdot\\frac{1}{1+2\\frac{e}{e_\\mathrm{tot}}}` is the neighborhood radius.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        density : typing.Union[np.ndarray, typing.Callable]
            ???.
        n_epochs : int, default: **None**
            Optional number of epochs to train for.
        n_vectors : int, default: **None**
            Optional number of vectors to train for.
        stop_quantization_error : float, default: **None**
            Stops training if quantization_error < stop_quantization_error.
        stop_topographic_error : float, default: **None**
            Stops training if topographic_error < stop_topographic_error.
        show_progress_bar : bool, default: **False**
            Specifies whether to display a progress bar.
        enable_gpu : bool, default: **True**
            If available, run on GPU rather than CPU.
        threads_per_blocks : int, default: **1024**
            Number of GPU threads per blocks.
        """

        if stop_quantization_error is None:
            stop_quantization_error = -1.0e6

        if stop_topographic_error is None:
            stop_topographic_error = -1.0e6

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        self._n_epochs = n_epochs

        self._n_vectors = n_vectors

        if not (n_epochs is None) and (n_vectors is None):

            ############################################################################################################
            # TRAINING BY NUMBER OF EPOCHS                                                                             #
            ############################################################################################################

            self._quantization_errors = np.full(n_epochs, np.nan, dtype = np.float32)

            self._topographic_errors = np.full(n_epochs, np.nan, dtype = np.float32)

            ############################################################################################################

            for cur_epoch in tqdm.trange(n_epochs, disable = not show_progress_bar):

                ########################################################################################################

                generator = generator_builder()

                for vectors in generator():

                    SOM_Online._train_step1_epoch(
                        self._weights,
                        self._topography,
                        vectors.astype(self._dtype),
                        cur_epoch,
                        n_epochs,
                        self._dtype(self._alpha),
                        self._dtype(self._sigma),
                        self._m * self._n
                    )

                    gc.collect()

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

            self._quantization_errors = np.zeros(1, dtype = np.float32)

            self._topographic_errors = np.zeros(1, dtype = np.float32)

            ############################################################################################################

            progress_bar = tqdm.tqdm(total = n_vectors, disable = not show_progress_bar)

            generator = generator_builder()

            for vectors in generator():

                count = min(vectors.shape[0], n_vectors - cur_vector)

                SOM_Online._train_step1_iter(
                    self._weights,
                    self._topography,
                    vectors[0: count].astype(self._dtype),
                    cur_vector,
                    n_vectors,
                    self._dtype(self._alpha),
                    self._dtype(self._sigma),
                    self._m * self._n
                )

                gc.collect()

                cur_vector += count

                progress_bar.update(count)

                if cur_vector >= n_vectors:

                    break

            ############################################################################################################

            errors = self.compute_errors(dataset, show_progress_bar = False, enable_gpu = enable_gpu, threads_per_blocks = threads_per_blocks)

            self._quantization_errors[0] = errors[0]
            self._topographic_errors[0] = errors[1]

            ############################################################################################################

        else:

            raise ValueError('Invalid training method, specify either `n_epochs` or `n_vectors`')

########################################################################################################################

@nb.njit(fastmath = True)
def _train_step2(weights: np.ndarray, topography: np.ndarray, vector: np.ndarray, alpha: float, sigma: float, mn: int) -> None:

    ####################################################################################################################
    # DO BMUS CALCULATION                                                                                              #
    ####################################################################################################################

    min_distance = 1.0e10
    min_index = 0

    for index in range(mn):

        distance = np.sum((weights[index] - vector) ** 2)

        if min_distance > distance:

            min_distance = distance
            min_index = index

    ####################################################################################################################

    bmu1 = topography[min_index]

    ####################################################################################################################
    # DO NEIGHBORHOOD OPERATOR CALCULATION                                                                             #
    ####################################################################################################################

    neighborhood_op = np.exp(-np.sum((topography - bmu1) ** 2, axis = -1) / (2.0 * sigma ** 2))

    ####################################################################################################################
    # UPDATE WEIGHTS                                                                                                   #
    ####################################################################################################################

    weights += alpha * np.expand_dims(neighborhood_op, -1) * (vector - weights)

########################################################################################################################
