# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb

from . import som_abstract, asymptotic_decay, dataset_to_generator_builder

########################################################################################################################

class SOM_Batch(som_abstract.SOM_Abstract):

    """
    Self Organizing Maps (standard batch implementation).
    """

    __MODE__ = 'batch'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[np.single] = np.float32, topology: typing.Optional[str] = None, alpha: float = None, sigma: float = None):

        """
        A rule of thumb to set the size of the grid for a dimensionality reduction
        task is that it should contain \\( 5\\sqrt{N} \\) neurons where N is the
        number of samples in the dataset to analyze.

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
