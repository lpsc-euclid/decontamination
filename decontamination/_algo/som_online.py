# -*- coding: utf-8 -*-
########################################################################################################################

import math
import tqdm
import typing

import numpy as np
import numba as nb

from . import abstract_som, asymptotic_decay, dataset_to_generator_builder

########################################################################################################################

class SOM_Online(abstract_som.AbstractSOM):

    """
    Self Organizing Maps (standard online implementation).
    """

    __MODE__ = 'online'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[np.single] = np.float32, topology: typing.Optional[str] = None, alpha: float = None, sigma: float = None):

        """
        Constructor for the Abstract Self Organizing Map (SOM).

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
        alpha : float
            Starting value of the learning rate (default: 0.3).
        sigma : float
            Starting value of the neighborhood radius (default: \\( \\mathrm{max}(m,n)/2 \\)).
        """

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology)

        ################################################################################################################

        self._alpha = 0.3 if alpha is None else dtype(alpha)

        self._sigma = max(m, n) / 2.0 if sigma is None else dtype(sigma)

        ################################################################################################################

        self._n_epochs = None

        self._n_vectors = None

        ################################################################################################################

        self._quantization_errors = None

        self._topographic_errors = None

    ####################################################################################################################

    def save(self, filename: str, **kwargs) -> None:

        """
        Saves the trained neural network to a file.

        Parameters
        ----------
        filename : str
            Output HDF5 filename.
        """

        super().save(filename, {
            'mode': '__MODE__',
            'alpha': '_alpha',
            'sigma': '_sigma',
            'n_epochs': '_n_epochs',
            'n_vectors': '_n_vectors',
        }, {
            'quantization_errors': '_quantization_errors',
            'topographic_errors': '_topographic_errors',
        })

    ####################################################################################################################

    def load(self, filename: str, **kwargs) -> None:

        """
        Loads the trained neural network from a file.

        Parameters
        ----------
        filename : str
            Input HDF5 filename.
        """

        super().load(filename, {
            'mode': '__MODE__',
            'alpha': '_alpha',
            'sigma': '_sigma',
            'n_epochs': '_n_epochs',
            'n_vectors': '_n_vectors',
        }, {
            'quantization_errors': '_quantization_errors',
            'topographic_errors': '_topographic_errors',
        })

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _train_step1_epoch(weights: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, topography: np.ndarray, data: np.ndarray, cur_epoch: int, n_epochs: int, alpha0: float, sigma0: float, mn: int):

        ################################################################################################################

        decay_function = asymptotic_decay(cur_epoch, n_epochs)

        alpha = alpha0 * decay_function

        sigma = sigma0 * decay_function

        ################################################################################################################

        for i in range(data.shape[0]):

            quantization_error, topographic_error = _train_step2(weights, topography, data[i], alpha, sigma, mn)

            quantization_errors[cur_epoch] += quantization_error
            topographic_errors[cur_epoch] += topographic_error

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _train_step1_iter(weights: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, topography: np.ndarray, data: np.ndarray, cur_vector: int, n_vectors: int, n_err_bins: int, alpha0: float, sigma0: float, mn: int):

        for i in range(data.shape[0]):

            ############################################################################################################

            decay_function = asymptotic_decay(cur_vector + i, n_vectors)

            alpha = alpha0 * decay_function

            sigma = sigma0 * decay_function

            ############################################################################################################

            quantization_error, topographic_error = _train_step2(weights, topography, data[i], alpha, sigma, mn)

            cur_err_bin = (n_err_bins * (cur_vector + i)) // n_vectors

            quantization_errors[cur_err_bin] += quantization_error
            topographic_errors[cur_err_bin] += topographic_error

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], n_epochs: typing.Optional[int] = None, n_vectors: typing.Optional[int] = None, n_error_bins: typing.Optional[int] = 10, show_progress_bar: bool = False) -> None:

        """
        Trains the neural network. Use either the `n_epochs` or `n_vectors` methods.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        n_epochs : typing.Optional[int]
            Number of epochs to train for (default: None).
        n_vectors : typing.Optional[int]
            Number of vectors to train for (default: None).
        n_error_bins : int
            Number of error bins (default: 10).
        show_progress_bar : bool
            Specifies whether to display a progress bar (default: **False**).
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        cur_vector = 0

        self._n_epochs = n_epochs

        self._n_vectors = n_vectors

        if not (n_epochs is None) and (n_vectors is None):

            ############################################################################################################
            # TRAINING BY NUMBER OF EPOCHS                                                                             #
            ############################################################################################################

            self._quantization_errors = np.zeros(n_epochs, dtype = np.float32)

            self._topographic_errors = np.zeros(n_epochs, dtype = np.float32)

            ############################################################################################################

            for cur_epoch in tqdm.trange(n_epochs, disable = not show_progress_bar):

                generator = generator_builder()

                for data in generator():

                    cur_vector += data.shape[0]

                    SOM_Online._train_step1_epoch(
                        self._weights,
                        self._quantization_errors,
                        self._topographic_errors,
                        self._topography,
                        data,
                        cur_epoch,
                        n_epochs,
                        self._alpha,
                        self._sigma,
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

            for data in generator():

                count = min(data.shape[0], n_vectors - cur_vector)

                SOM_Online._train_step1_iter(
                    self._weights,
                    self._quantization_errors,
                    self._topographic_errors,
                    self._topography,
                    data[0: count],
                    cur_vector,
                    n_vectors,
                    n_error_bins,
                    self._alpha,
                    self._sigma,
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

            raise Exception('Invalid training method, use either `n_epochs` or `n_vectors`.')

########################################################################################################################

@nb.njit(parallel = False, fastmath = True)
def _train_step2(weights: np.ndarray, topography: np.ndarray, vector: np.ndarray, alpha: float, sigma: float, mn: int):

    ####################################################################################################################
    # BMUS CALCULATION                                                                                                 #
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
    # LEARNING OPERATOR CALCULATION                                                                                    #
    ####################################################################################################################

    distance_matrix = np.sum((topography - bmu1) ** 2, axis = -1)

    learning_op = alpha * np.exp(-distance_matrix / (2.0 * sigma ** 2))

    ####################################################################################################################
    # UPDATE WEIGHTS                                                                                                   #
    ####################################################################################################################

    weights += np.expand_dims(learning_op, axis = -1) * (vector - weights)

    ####################################################################################################################
    # UPDATE ERRORS                                                                                                    #
    ####################################################################################################################

    if np.sum((bmu1 - bmu2) ** 2) > 2:

        return math.sqrt(min_distance1), 1

    return math.sqrt(min_distance1), 0

########################################################################################################################
