# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import gc
import tqdm
import typing

import numpy as np
import numba as nb

from . import som_abstract, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming, DuplicatedCode
class SOM_PCA(som_abstract.SOM_Abstract):

    """
    Self Organizing Maps that span the first two principal components. It runs with constant memory usage.

    .. note::
        A rule of thumb to set the size of the grid for a dimensionality reduction task is that it should contain :math:`5\\sqrt{N}` neurons where N is the number of samples in the dataset to analyze.

    Parameters
    ----------
    m : int
        Number of neuron rows.
    n : int
        Number of neuron columns.
    dim : int
        Dimensionality of the input data.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]], default: **np.float32**
        Neural network data type, either **np.float32** or **np.float64**.
    topology : str, default: **None** ≡ **'hexagonal'**
        Neural network topology, either **'square'** or **'hexagonal'**.
    """

    __MODE__ = 'pca'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, topology: typing.Optional[str] = None):

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology)

        ################################################################################################################

        self._cov_matrix = None

        self._eigenvalues = None

        self._eigenvectors = None

        self._orders = None

        ################################################################################################################

        self._header_extra = {
            'mode': '__MODE__',
        }

    ####################################################################################################################

    @property
    def cov_matrix(self) -> np.ndarray:

        """Covariance matrix of the training dataset."""

        return self._cov_matrix

    ####################################################################################################################

    @property
    def eigenvalues(self) -> np.ndarray:

        """Eigenvalues of the covariance matrix."""

        return self._eigenvalues

    ####################################################################################################################

    @property
    def eigenvectors(self) -> np.ndarray:

        """Eigenvectors of the covariance matrix."""

        return self._eigenvectors

    ####################################################################################################################

    @property
    def orders(self) -> np.ndarray:

        """Order of importance of the components."""

        return self._orders

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _update_cov_matrix(result_sum: np.ndarray, result_prods: np.ndarray, vectors: np.ndarray, density: np.ndarray) -> int:

        ################################################################################################################

        n = 0

        data_dim = vectors.shape[0]
        syst_dim = vectors.shape[1]

        ################################################################################################################

        for i in range(data_dim):

            vector = vectors[i] \
                        .astype(np.float64)
            weight = density[i]

            if not np.any(np.isnan(vector)):

                n += weight

                for j in range(syst_dim):

                    vector_j = weight * vector[j]
                    result_sum[j] += vector_j

                    for k in range(syst_dim):

                        vector_jk = vector_j * vector[k]
                        result_prods[j][k] += vector_jk

        ################################################################################################################

        return n

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _diag_cov_matrix(weights: np.ndarray, cov_matrix: np.ndarray, min_weight: float, max_weight: float, m: int, n: int) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

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

        ################################################################################################################

        return eigenvalues, eigenvectors, orders

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], density: typing.Optional[typing.Union[np.ndarray, typing.Callable]] = None, min_weight: float = 0.0, max_weight: float = 1.0, show_progress_bar: bool = False) -> None:

        """
        Trains the neural network.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        density : typing.Union[np.ndarray, typing.Callable]
            ???.
        min_weight : float, default: **0.0**
            Latent space minimum value.
        max_weight : float, default: **1.0**
            Latent space maximum value.
        show_progress_bar : bool, default: **False**
            Specifies whether to display a progress bar.
        """

        ################################################################################################################

        dateset_generator_builder = dataset_to_generator_builder(dataset)
        density_generator_builder = dataset_to_generator_builder(density)

        ################################################################################################################

        total_nb = 0

        total_sum = np.zeros((self._dim, ), dtype = np.float64)
        total_prods = np.zeros((self._dim, self._dim, ), dtype = np.float64)

        ################################################################################################################

        if density_generator_builder is not None:

            dateset_generator = dateset_generator_builder()
            density_generator = density_generator_builder()

            for vectors, density in tqdm.tqdm(zip(dateset_generator(), density_generator()), disable = not show_progress_bar):

                total_nb += SOM_PCA._update_cov_matrix(
                    total_sum,
                    total_prods,
                    vectors,
                    density.astype(np.int64)
                )

                gc.collect()

        else:

            dateset_generator = dateset_generator_builder()

            for vectors in tqdm.tqdm(dateset_generator(), disable = not show_progress_bar):

                total_nb += SOM_PCA._update_cov_matrix(
                    total_sum,
                    total_prods,
                    vectors,
                    np.ones(vectors.shape[0], dtype = np.int64)
                )

                gc.collect()

        ################################################################################################################

        total_sum /= total_nb
        total_prods /= total_nb

        ################################################################################################################

        self._cov_matrix = total_prods - np.outer(total_sum, total_sum)

        ################################################################################################################

        self._eigenvalues, self._eigenvectors, self._orders = SOM_PCA._diag_cov_matrix(
            self.centroids,
            self.cov_matrix,
            min_weight,
            max_weight,
            self._m,
            self._n
        )

########################################################################################################################
