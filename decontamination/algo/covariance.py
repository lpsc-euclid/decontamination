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

from . import dataset_to_generator_builder

########################################################################################################################

class Covariance(object):

    """
    Covariance calculation with constant memory usage.
    """

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _update_welford_cov_sums(sum_w: float, mean: np.ndarray, m2_upper: np.ndarray, delta: np.ndarray, vectors: np.ndarray, weights: np.ndarray, dim: int) -> float:

        ################################################################################################################

        for i in range(vectors.shape[0]):

            ############################################################################################################

            w = weights[i]

            if (not np.isfinite(w)) or (w <= 0.0):

                continue

            x = vectors[i]

            ############################################################################################################

            if np.all(np.isfinite(x)):

                ########################################################################################################

                sum_w_new = sum_w + w

                ########################################################################################################

                for d in range(dim):

                    delta[d] = x[d] - mean[d]

                ########################################################################################################

                a = w / sum_w_new

                for d in range(dim):

                    mean[d] += a * delta[d]

                ########################################################################################################

                for j in range(dim):

                    dj = delta[j]

                    for k in range(j, dim):

                        m2_upper[j, k] += w * dj * (x[k] - mean[k])

                ########################################################################################################

                sum_w = sum_w_new

        ################################################################################################################

        return sum_w

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _finalize_welford_cov(m2_upper: np.ndarray, sum_w: float, dim: int) -> np.ndarray:

        ################################################################################################################

        cov_matrix = np.empty((dim, dim), dtype = np.float64)

        ################################################################################################################

        inv_sum_w = 1.0 / sum_w

        ################################################################################################################

        for j in range(dim):

            cov_matrix[j, j] = m2_upper[j, j] * inv_sum_w

            for k in range(j + 1, dim):

                v = m2_upper[j, k] * inv_sum_w

                cov_matrix[j, k] = v
                cov_matrix[k, j] = v

        ################################################################################################################

        return cov_matrix

    ####################################################################################################################

    @staticmethod
    def compute(dim: int, dataset: typing.Union[np.ndarray, typing.Callable], dataset_weights: typing.Optional[typing.Union[np.ndarray, typing.Callable]] = None, show_progress_bar: bool = False):

        """
        Computes the covariance matrix of the given dataset using Welford's algorithm.

        Parameters
        ----------
        dim : int
            Dimensionality of the input data.
        dataset : typing.Union[np.ndarray, typing.Callable]
            Dataset array or generator builder.
        dataset_weights : typing.Union[np.ndarray, typing.Callable], default: **None**
            Dataset weight array or generator builder.
        show_progress_bar : bool, default: **False**
            Specifies whether to display a progress bar.

        Returns
        -------
        np.ndarray
            The covariance matrix of the given dataset.
        """

        ################################################################################################################

        dataset_generator_builder = dataset_to_generator_builder(    dataset    )
        weight_generator_builder = dataset_to_generator_builder(dataset_weights)

        ################################################################################################################

        total_w = 0.0

        mean = np.zeros((dim, ), dtype = np.float64)
        m2_upper = np.zeros((dim, dim, ), dtype = np.float64)
        delta = np.empty((dim, ), dtype = np.float64)

        ################################################################################################################

        if weight_generator_builder is not None:

            dataset_generator = dataset_generator_builder()
            weight_generator = weight_generator_builder()

            for vectors, weights in tqdm.tqdm(zip(dataset_generator(), weight_generator()), disable = not show_progress_bar):

                total_w = Covariance._update_welford_cov_sums(
                    total_w,
                    mean,
                    m2_upper,
                    delta,
                    vectors.astype(np.float64, copy = False),
                    weights.astype(np.float64, copy = False),
                    dim
                )

                gc.collect()

        else:

            dataset_generator = dataset_generator_builder()

            for vectors in tqdm.tqdm(dataset_generator(), disable = not show_progress_bar):

                total_w = Covariance._update_welford_cov_sums(
                    total_w,
                    mean,
                    m2_upper,
                    delta,
                    vectors.astype(np.float64, copy = False),
                    np.ones(vectors.shape[0], dtype = np.float64),
                    dim
                )

                gc.collect()

        ################################################################################################################

        if total_w > 0.0:

            return Covariance._finalize_welford_cov(m2_upper, total_w, dim)

        else:

            raise ValueError('Empty dataset or total weight is zero.')

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def diagonalize(cov_matrix: np.ndarray, sort: bool = True) -> typing.Tuple[np.ndarray, np.ndarray, typing.Optional[np.ndarray]]:

        """
        Diagonalizes the given covariance matrix.

        Parameters
        ----------
        cov_matrix : np.ndarray
            The covariance matrix to be diagonalized.
        sort: bool, default: **True**
            If **True**, eigenvalues and eigenvectors are sorted by decreasing eigenvalue.

        Returns
        -------
        np.ndarray
            The eigenvalues.
        np.ndarray
            The eigenvectors.
        typing.Optional[np.ndarray]
            If `sort` is **True**, order of importance of the components, else **None**.
        """

        ################################################################################################################

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        ################################################################################################################

        if not sort:

            orders = None

        else:

            orders = np.argsort(eigenvalues)[:: -1]

            eigenvalues = np.take(eigenvalues, orders, axis = 0)
            eigenvectors = np.take(eigenvectors, orders, axis = 1)

        ################################################################################################################

        return eigenvalues, eigenvectors, orders

########################################################################################################################
