# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np
import numba as nb

from . import abstract_som

########################################################################################################################

class SOM_Batch(abstract_som.AbstractSOM):

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: np.dtype = np.float32, topology = 'hexagonal'):

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
        topology : Optional[str]
            Topology of the map, '**square**' or '**hexagonal**' (default: '**hexagonal**').
        """

        super().__init__(m, n, dim, dtype, topology)

########################################################################################################################
