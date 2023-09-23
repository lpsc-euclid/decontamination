# -*- coding: utf-8 -*-
########################################################################################################################

import tqdm
import typing

import numpy as np
import numba as nb

from . import abstract_som, dataset_to_generator_builder

########################################################################################################################

class SOM_PCA(abstract_som.AbstractSOM):

    """
    Self Organizing Maps that span the first two principal components.
    """

    __MODE__ = 'pca'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[np.single] = np.float32, topology: typing.Optional[str] = None):

        """
        Initializes a Self Organizing Maps.

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
        """

        super().__init__(m, n, dim, dtype, topology)

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
        })

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _cov_matrix_kernel(result_sum: np.ndarray, result_prods: np.ndarray, data: np.ndarray) -> None:

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

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], show_progress_bar: bool = False) -> None:

        """
        Trains the neural network.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator of generator.
        show_progress_bar : bool
            Specifies whether to display a progress bar (default: **False**).
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        generator = generator_builder()

        ################################################################################################################

        total_nb = 0

        total_sum = np.zeros((self._dim, ), dtype = np.float64)
        total_prods = np.zeros((self._dim, self._dim, ), dtype = np.float64)

        ################################################################################################################

        for data in tqdm.tqdm(generator(), disable = not show_progress_bar):

            SOM_PCA._cov_matrix_kernel(total_sum, total_prods, data)

            total_nb += data.shape[0]

        ################################################################################################################

        total_sum /= total_nb
        total_prods /= total_nb

        ################################################################################################################

        cov_matrix = total_prods - np.outer(total_sum, total_sum)

        ################################################################################################################

        SOM_PCA._diag_kernel(self.get_centroids(), cov_matrix, self._m, self._n)

########################################################################################################################
