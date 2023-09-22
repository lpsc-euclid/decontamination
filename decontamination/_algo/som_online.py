# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb
import tqdm

from . import abstract_som, asymptotic_decay, dataset_to_generator_builder

########################################################################################################################

class SOM_Online(abstract_som.AbstractSOM):

    """
    Self Organizing Maps (standard online implementation).
    """

    __MODE__ = 'online'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[np.single] = np.float32, topology: typing.Optional[str] = None, alpha: float = None, sigma: float = None, decay_function = asymptotic_decay):

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
        decay_function : function
            Function that reduces alpha and sigma at each iteration (default: \\( 1/\\left(1+2\\frac{epoch}{epochs}\\right) \\)).
        """

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology)

        ################################################################################################################

        self._epochs = 0

        self._decay_function = decay_function

        self._alpha = 0.3 if alpha is None else dtype(alpha)

        self._sigma = max(m, n) / 2.0 if sigma is None else dtype(sigma)

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
        }, {

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
        }, {

        })

    ####################################################################################################################

    @staticmethod
    # @nb.njit
    def _train(weights: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, topography: np.ndarray, data: np.ndarray, alpha: float, sigma: float, m: int, n: int):

        for i in range(data.shape[0]):

            vector = data[i]

            ############################################################################################################
            # BMUS CALCULATION                                                                                         #
            ############################################################################################################

            min_dist = 1.0e99
            min_index1 = 0
            min_index2 = 0

            for cur_index in range(m * n):

                dist = np.sum((weights[cur_index] - vector) ** 2)

                if min_dist > dist:

                    min_index2 = min_index1
                    min_index1 = cur_index
                    min_dist = dist

            ############################################################################################################
            # NEIGHBORHOOD CALCULATION                                                                                 #
            ############################################################################################################

            bmu_distance_squares = np.sum((topography - topography[min_index1]) ** 2, axis = -1)

            neighbourhood_op = np.exp(- bmu_distance_squares / (2.0 * np.square(sigma)))

            ############################################################################################################
            # UPDATE WEIGHTS                                                                                           #
            ############################################################################################################

            weights += alpha * np.expand_dims(neighbourhood_op, axis =-1) * (vector - weights)

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], epochs: int = 1, show_progress_bar: bool = True) -> None:


        """
        Trains the neural network.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator of generator.
        epochs : int
            Number of training epochs (default: 1).
        show_progress_bar : bool
            Specifying whether a progress bar have to be shown (default: **True**).
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        generator = generator_builder()

        ################################################################################################################

        for epoch in tqdm.trange(epochs, disable = not show_progress_bar):

            decay_function = self._decay_function(epoch, epochs)

            alpha = self._alpha * decay_function

            sigma = self._sigma * decay_function

            for data in tqdm.tqdm(generator(), disable = not show_progress_bar):

                SOM_Online._train(self._weights, self._quantization_errors, self._topographic_errors, self._topography, data, alpha, sigma, self._m, self._n)

########################################################################################################################
