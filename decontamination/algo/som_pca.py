# -*- coding: utf-8 -*-
########################################################################################################################

import tqdm
import typing

import numpy as np
import numba as nb

from . import abstract_som, dataset_to_generator_of_generator

########################################################################################################################

class SOM_PCA(abstract_som.AbstractSOM):

    """
    Self Organizing Maps that span the first two principal components.
    """

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: np.dtype = np.float32, topology: typing.Optional[str] = None):

        """
        Initializes a Self Organizing Maps.

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
            dtype : np.dtype
                Neural network data type (default: **np.float32**).
            topology : Optional[str]
                Topology of the map, **square** or **hexagonal** (default: **hexagonal**).
        """

        super().__init__(m, n, dim, dtype, topology)

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _cov_matrix_kernel(result_sum: np.ndarray, result_prods: np.ndarray, data: np.ndarray, data_dim: int, syst_dim: int) -> None:

        for i in range(data_dim):

            value = data[i].astype(np.float64)

            for j in range(syst_dim):

                value_j = value[j]
                result_sum[j] += value_j

                for k in range(syst_dim):

                    value_jk = value_j * value[k]
                    result_prods[j][k] += value_jk

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _diag_kernel(weights: np.ndarray, cov_matrix: np.ndarray, m: int, n: int) -> None:

        ################################################################################################################

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        orders = np.argsort(-eigenvalues)

        order0 = orders[0]
        order1 = orders[1]

        ################################################################################################################

        linspace_x = np.linspace(-1, 1, m)
        linspace_y = np.linspace(-1, 1, n)

        for i in range(m):
            c1 = linspace_x[i]

            for j in range(n):
                c2 = linspace_y[j]

                weights[i, j] = (
                    eigenvectors[:, order0] * c1
                    +
                    eigenvectors[:, order1] * c2
                ).astype(weights.dtype)

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], show_progress_bar: bool = True) -> None:

        """
        Trains the neural network.

        Parameters
        ----------
        dataset : typing.Union[numpy.ndarray, typing.Callable]
            Training dataset array or generator of generator.
        show_progress_bar : bool
            Specifying whether a progress bar have to be shown (default: **True**).
        """

        ################################################################################################################

        generator_of_generator = dataset_to_generator_of_generator(dataset)

        generator = generator_of_generator()

        ################################################################################################################

        total_nb = 0

        total_sum = np.zeros((self._dim, ), dtype = np.float64)
        total_prods = np.zeros((self._dim, self._dim, ), dtype = np.float64)

        ################################################################################################################

        for data in tqdm.tqdm(generator(), disable = not show_progress_bar):

            total_nb += data.shape[0]

            sub_sum = np.zeros_like(total_sum)
            sub_prods = np.zeros_like(total_prods)

            SOM_PCA._cov_matrix_kernel(sub_sum, sub_prods, data, data.shape[0], data.shape[1])

            total_sum += sub_sum
            total_prods += sub_prods

        ################################################################################################################

        total_sum /= total_nb
        total_prods /= total_nb

        ################################################################################################################

        cov_matrix = total_prods - np.outer(total_sum, total_sum)

        ################################################################################################################

        SOM_PCA._diag_kernel(self.get_centroids(), cov_matrix, self._m, self._n)

########################################################################################################################
