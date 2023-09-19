# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb

from . import dataset_to_generator_of_generator

########################################################################################################################

class PCA(object):

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: type = np.float32):

        """
        Initializes a Self Organizing Maps to span the first two principal components.

        A rule of thumb to set the size of the grid for a dimensionality reduction
        task is that it should contain \\( 5\\sqrt{N} \\) neurons where N is the
        number of samples in the dataset to analyze.

        Arguments
        ---------
        m : int
            Number of neuron rows.
        n : int
            Number of neuron columns.
        dim : int
            Dimensionality of the input data.
        dtype : type
            Neural network data type (default: **np.float32**).
        """

        self._m = m
        self._n = n
        self._dim = dim
        self._dtype = dtype

    ####################################################################################################################

    @staticmethod
    def _cov_matrix_kernel():

        pass

    ####################################################################################################################

    @staticmethod
    def _diag_kernel():

        pass

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], show_progress_bar: bool = True) -> None:

        """
        Trains the neural network.

        Parameters
        ----------
        dataset : typing.Union[numpy.ndarray, typing.Callable]
            Training dataset array or generator.
        show_progress_bar : bool
            Specifying whether a progress bar have to be shown (default: **True**).
        """

        generator_of_generator = dataset_to_generator_of_generator(dataset)

        generator = generator_of_generator()

########################################################################################################################
