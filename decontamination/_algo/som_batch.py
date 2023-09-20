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

    def __init__(self, m: int, n: int, dim: int, dtype: np.dtype = np.float32, topology: typing.Optional[str] = None, seed: int = None, alpha: float = None, sigma: float = None, decay_function = asymptotic_decay):

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
        dtype : np.dtype
            Neural network data type (default: **np.float32**).
        topology : typing.Optional[str]
            Topology of the map, either '**square**' or '**hexagonal**' (default: '**hexagonal**').
        seed : int
            Seed for random generator (default: **None**).
        alpha : float
            Starting value of the learning rate (default: 0.3).
        sigma : float
            Starting value of the neighborhood radius (default: \\( \\mathrm{max}(m,n)/2 \\)).
        decay_function : function
            Function that reduces alpha and sigma at each iteration (default: \\( 1/\\left(1+2\\frac{epoch}{epochs}\\right) \\)).
        """

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology, seed)

        ################################################################################################################

        self._epochs = 0

        self._decay_function = decay_function

        self._alpha = 0.3 if alpha is None else dtype(alpha)

        self._sigma = max(m, n) / 2.0 if sigma is None else dtype(sigma)

    ####################################################################################################################

    def save(self, filename: str) -> None:

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

    def load(self, filename: str) -> None:

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

########################################################################################################################
