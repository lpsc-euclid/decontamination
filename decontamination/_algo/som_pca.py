# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb

from . import som_abstract, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class SOM_PCA(som_abstract.SOM_Abstract):

    """
    Self Organizing Maps that span the first two principal components.
    """

    __MODE__ = 'pca'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[np.single] = np.float32, topology: typing.Optional[str] = None):

        """
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
        """

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology)

        ################################################################################################################

        self._header_extra = {
            'mode': '__MODE__',
        }

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _update_cov_matrix(result_sum: np.ndarray, result_prods: np.ndarray, data: np.ndarray) -> None:

        ################################################################################################################

        data_dim = data.shape[0]
        syst_dim = data.shape[1]

        ################################################################################################################

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
    def _diag_cov_matrix(weights: np.ndarray, cov_matrix: np.ndarray, min_weight: float, max_weight: float, m: int, n: int) -> None:

        ################################################################################################################

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        orders = np.argsort(-eigenvalues)

        order0 = orders[0]
        order1 = orders[1]

        ################################################################################################################

        linspace_x = np.linspace(min_weight, max_weight, m)
        linspace_y = np.linspace(min_weight, max_weight, n)

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

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], min_weight: float = 0.0, max_weight: float = 1.0) -> None:

        """
        Trains the neural network.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        min_weight : float
            Minimum value in the latent space (default: **O.O**)
        max_weight : float
            Maximum value in the latent space (default: **1.O**)
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        generator = generator_builder()

        ################################################################################################################

        total_nb = 0

        total_sum = np.zeros((self._dim, ), dtype = np.float64)
        total_prods = np.zeros((self._dim, self._dim, ), dtype = np.float64)

        ################################################################################################################

        for data in generator():

            total_nb += data.shape[0]

            SOM_PCA._update_cov_matrix(total_sum, total_prods, data)

        ################################################################################################################

        total_sum /= total_nb
        total_prods /= total_nb

        ################################################################################################################

        cov_matrix = total_prods - np.outer(total_sum, total_sum)

        ################################################################################################################

        SOM_PCA._diag_cov_matrix(self.get_centroids(), cov_matrix, min_weight, max_weight, self._m, self._n)

########################################################################################################################
