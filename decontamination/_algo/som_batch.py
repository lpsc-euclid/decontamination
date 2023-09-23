# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb

from . import abstract_som, asymptotic_decay, dataset_to_generator_builder

########################################################################################################################

class SOM_Batch(abstract_som.AbstractSOM):

    """
    Self Organizing Maps (standard batch implementation).
    """

    __MODE__ = 'batch'

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

        pass

########################################################################################################################
