# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb

from . import abstract_som, dataset_to_generator_of_generator

########################################################################################################################

class PCA(abstract_som.AbstractSOM):

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

        super().__init__(m, n, dim, dtype)

    ####################################################################################################################

    @staticmethod
    @nb.njit
    def _cov_matrix_kernel(batch_arr):

        Ndata = batch_arr.shape[0]
        Nsyst = batch_arr.shape[1]
        summ = np.zeros((Nsyst, ))
        prod = np.zeros((Nsyst, Nsyst))

        for data_index in nb.prange(Ndata):

            data = batch_arr[data_index]

            for i_syst in range(Nsyst):

                summ[i_syst] += data[i_syst]

                for j_syst in range(Nsyst):

                    prod[i_syst][j_syst] += data[i_syst] * data[j_syst]

        return Ndata, summ, prod


    ####################################################################################################################

    @staticmethod
    @nb.njit
    def _diag_kernel(weights, cov, m, n):

        pc_length, pc = np.linalg.eig(cov)

        pc_order = np.argsort(-pc_length)

        C1 = np.repeat(np.linspace(-1, 1, n), m).astype(np.float64)
        C2 = np.repeat(np.linspace(-1, 1, m), n).reshape(-1, n).T.ravel().astype(np.float64)

        weights[:] = np.expand_dims(C1, axis = -1) * pc[:, pc_order[0]] + np.expand_dims(C2, axis = -1) * pc[:, pc_order[1]]

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

        num_samples = 0
        sum_values = np.zeros((self._dim, ), dtype = self._dtype)
        sum_products = np.zeros((self._dim, self._dim), dtype = self._dtype)

        for batch in generator():

            num_sample, sum_value, sum_product = PCA._cov_matrix_kernel(batch)

            num_samples += num_sample
            sum_values += sum_value
            sum_products += sum_product

        mean_values = sum_values / num_samples

        cov = (sum_products / num_samples) - np.outer(mean_values, mean_values)

        PCA._diag_kernel(self._weights, cov, self._m, self._n)

########################################################################################################################
