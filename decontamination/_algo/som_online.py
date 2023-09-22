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

        self._epochs = None

        self._n_max_vectors = None

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
            'epochs': '_epochs',
            'n_max_vectors': '_n_max_vectors',
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
            'epochs': '_epochs',
            'n_max_vectors': '_n_max_vectors',
        }, {
            'quantization_errors': '_quantization_errors',
            'topographic_errors': '_topographic_errors',
        })

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _train_step1_epoch(weights: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, topography: np.ndarray, data: np.ndarray, epoch: int, epochs: int, alpha0: float, sigma0: float, m: int, n: int):

        ################################################################################################################

        decay_function = asymptotic_decay(epoch, epochs)

        alpha = alpha0 * decay_function

        sigma = sigma0 * decay_function

        ################################################################################################################

        for i in range(data.shape[0]):

            quantization_error, topographic_error = _train_step2(weights, topography, data[i], alpha, sigma, m, n)

            quantization_errors[epoch] += quantization_error
            topographic_errors[epoch] += topographic_error

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _train_step1_iter(weights: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, topography: np.ndarray, data: np.ndarray, n_vectors: int, n_max_vectors: int, n_err_bins: int, alpha0: float, sigma0: float, m: int, n: int):

        for i in range(data.shape[0]):

            ############################################################################################################

            decay_function = asymptotic_decay(n_vectors + i, n_max_vectors)

            alpha = alpha0 * decay_function

            sigma = sigma0 * decay_function

            ############################################################################################################

            quantization_error, topographic_error = _train_step2(weights, topography, data[i], alpha, sigma, m, n)

            err_bin = math.floor(n_err_bins * (n_vectors + i) / n_max_vectors)

            quantization_errors[err_bin] += quantization_error
            topographic_errors[err_bin] += topographic_error

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], epochs: typing.Optional[int] = None, n_max_vectors: typing.Optional[int] = None, n_error_bins: typing.Optional[int] = 10, show_progress_bar: bool = True) -> None:

        """
        Trains the neural network.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        epochs : typing.Optional[int]
            ???
        n_max_vectors : typing.Optional[int]
            ???
        n_error_bins : int
            ???
        show_progress_bar : bool
            Specifying whether a progress bar have to be shown (default: **True**).
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        n_vectors = 0

        self._epochs = epochs

        self._n_max_vectors = n_max_vectors

        if not (epochs is None) and (n_max_vectors is None):

            ############################################################################################################
            # TRAINING BY NUMBER OF EPOCHS                                                                             #
            ############################################################################################################

            self._quantization_errors = np.zeros(epochs, dtype = np.float32)

            self._topographic_errors = np.zeros(epochs, dtype = np.float32)

            ############################################################################################################

            for epoch in tqdm.trange(epochs, disable = not show_progress_bar):

                generator = generator_builder()

                for data in generator():

                    n_vectors += data.shape[0]

                    SOM_Online._train_step1_epoch(
                        self._weights,
                        self._quantization_errors,
                        self._topographic_errors,
                        self._topography,
                        data,
                        epoch,
                        epochs,
                        self._alpha,
                        self._sigma,
                        self._m,
                        self._n
                    )

            ############################################################################################################

            if n_vectors > 0:

                self._quantization_errors = self._quantization_errors * epochs / n_vectors

                self._topographic_errors = self._topographic_errors * epochs / n_vectors

            ############################################################################################################

        elif (epochs is None) and not (n_max_vectors is None):

            ############################################################################################################
            # TRAINING BY NUMBER OF VECTORS                                                                            #
            ############################################################################################################

            self._quantization_errors = np.zeros(n_error_bins, dtype = np.float32)

            self._topographic_errors = np.zeros(n_error_bins, dtype = np.float32)

            ############################################################################################################

            generator = generator_builder()

            for data in generator():

                count = min(data.shape[0], n_max_vectors - n_vectors)

                SOM_Online._train_step1_iter(
                    self._weights,
                    self._quantization_errors,
                    self._topographic_errors,
                    self._topography,
                    data[0: count],
                    n_vectors,
                    n_max_vectors,
                    n_error_bins,
                    self._alpha,
                    self._sigma,
                    self._m,
                    self._n
                )

                n_vectors += count

                if n_vectors >= n_max_vectors:

                    break

            ############################################################################################################

            if n_vectors > 0:

                self._quantization_errors = self._quantization_errors * n_error_bins / n_vectors

                self._topographic_errors = self._topographic_errors * n_error_bins / n_vectors

            ############################################################################################################

        else:

            raise Exception('Invalid training method, use either `epochs` or `n_max_vectors`.')

########################################################################################################################

@nb.njit(parallel = False)
def _train_step2(weights: np.ndarray, topography: np.ndarray, vector: np.ndarray, alpha: float, sigma: float, m: int, n: int):

    ####################################################################################################################
    # BMUS CALCULATION                                                                                                 #
    ####################################################################################################################

    min_distance = 1.0e99

    min_index2 = 0
    min_index1 = 0

    for min_index0 in range(m * n):

        cur_distance = np.sum((weights[min_index0] - vector) ** 2)

        if min_distance > cur_distance:

            min_distance = cur_distance

            min_index2 = min_index1
            min_index1 = min_index0

    ####################################################################################################################
    # NEIGHBORHOOD CALCULATION                                                                                         #
    ####################################################################################################################

    bmu_distance_squares = np.sum((topography - topography[min_index1]) ** 2, axis = -1)

    neighbourhood_op = alpha * np.exp(-bmu_distance_squares / (2.0 * sigma ** 2))

    ####################################################################################################################
    # UPDATE WEIGHTS                                                                                                   #
    ####################################################################################################################

    weights += np.expand_dims(neighbourhood_op, axis = -1) * (vector - weights)

    ####################################################################################################################

    if np.sum((topography[min_index1] - topography[min_index2]) ** 2) > 2:

        return math.sqrt(min_distance), 1

    return math.sqrt(min_distance), 0

########################################################################################################################
