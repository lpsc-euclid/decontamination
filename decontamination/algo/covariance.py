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
    def _finalize_welford_cov(m2_upper: np.ndarray, sum_w: float, dim: int, ddof: int) -> np.ndarray:

        ################################################################################################################

        norm = sum_w - ddof

        if norm <= 0.0:

            raise ValueError('Empty dataset or degrees of freedom <= 0.')

        ################################################################################################################

        cov_matrix = np.empty((dim, dim), dtype = np.float64)

        ################################################################################################################

        for j in range(dim):

            cov_matrix[j, j] = m2_upper[j, j] / norm

            for k in range(j + 1, dim):

                v = m2_upper[j, k] / norm

                cov_matrix[j, k] = v
                cov_matrix[k, j] = v

        ################################################################################################################

        return cov_matrix

    ####################################################################################################################

    @staticmethod
    def compute(dim: int, dataset: typing.Union[np.ndarray, typing.Callable], dataset_weights: typing.Optional[typing.Union[np.ndarray, typing.Callable]] = None, ddof: int = 1, show_progress_bar: bool = False):

        """
        Computes the covariance matrix of the given dataset using Welford's algorithm.

        Parameters
        ----------
        dim : int
            Dimensionality of the input data.
        dataset : typing.Union[np.ndarray, typing.Callable]
            Dataset array of shape :math:`(N_\\mathrm{samples},\\mathrm{dim})` or generator builder.
        dataset_weights : typing.Union[np.ndarray, typing.Callable], default: **None**
            Dataset weight array of shape :math:`(N_\\mathrm{samples},)` or generator builder.
        ddof : int, default: **1**
            Delta degrees of freedom. Controls the bias of the estimate; normalization is :math:`N-\\mathrm{ddof}`.
        show_progress_bar : bool, default: **False**
            Specifies whether to display a progress bar.

        Returns
        -------
        np.ndarray
            The covariance matrix of the given dataset.
        """

        ################################################################################################################

        if ddof < 0:

            raise ValueError('ddof must be greater than or equal to zero.')

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

            for vectors_chunk, weights_chunk in tqdm.tqdm(zip(dataset_generator(), weight_generator()), disable = not show_progress_bar):

                if vectors_chunk.shape[1] != weights_chunk.shape[0]:

                    raise ValueError('`dataset` and `dataset_weights` chunks must be aligned')

                total_w = Covariance._update_welford_cov_sums(
                    total_w,
                    mean,
                    m2_upper,
                    delta,
                    vectors_chunk.astype(np.float64, copy = False),
                    weights_chunk.astype(np.float64, copy = False),
                    dim
                )

                gc.collect()

        else:

            dataset_generator = dataset_generator_builder()

            for vectors_chunk in tqdm.tqdm(dataset_generator(), disable = not show_progress_bar):

                total_w = Covariance._update_welford_cov_sums(
                    total_w,
                    mean,
                    m2_upper,
                    delta,
                    vectors_chunk.astype(np.float64, copy = False),
                    np.ones(vectors_chunk.shape[0], dtype = np.float64),
                    dim
                )

                gc.collect()

        ################################################################################################################

        return Covariance._finalize_welford_cov(m2_upper, total_w, dim, ddof)

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def diagonalize(cov_matrix: np.ndarray, sort: bool = True) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Diagonalizes the given covariance matrix.

        Parameters
        ----------
        cov_matrix : np.ndarray
            The covariance matrix to be diagonalized.
        sort : bool, default: **True**
            If **True**, eigenvalues and eigenvectors are sorted by decreasing eigenvalue.

        Returns
        -------
        np.ndarray
            The eigenvalues. If **sort** is **True**, eigenvalues are sorted.
        np.ndarray
            The eigenvectors. If **sort** is **True**, eigenvectors are sorted.
        np.ndarray
            If **sort** is **True**, order of importance of the components, else :math:`[0,1,\\dots,\\mathrm{dim}-1]`.
        """

        if cov_matrix.shape[0] != cov_matrix.shape[1]:

            raise ValueError('Input covariance matrix is not square.')

        ################################################################################################################

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        ################################################################################################################

        if sort:

            orders = np.argsort(eigenvalues)[:: -1]

            eigenvalues = np.take(eigenvalues, orders, axis = 0)
            eigenvectors = np.take(eigenvectors, orders, axis = 1)

        else:

            orders = np.arange(cov_matrix.shape[0], dtype = np.int64)

        ################################################################################################################

        return eigenvalues, eigenvectors, orders

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def project_pca(dataset: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:

        """
        Projects a dataset onto the Principal Component Analysis (PCA) basis.

        Parameters
        ----------
        dataset : np.ndarray
            Input dataset of shape :math:`(N_\\mathrm{samples},\\mathrm{dim})`.
        eigenvectors : np.ndarray
            Eigenvector matrix of shape :math:`(\\mathrm{dim},\\mathrm{dim})`.

        Returns
        -------
        np.ndarray
            The dataset projected onto the PCA basis :math:`\\equiv\\mathrm{dataset}\\cdot\\left(\\mathrm{eigenvectors}^{-1}\\right)^T`.
        """

        return np.dot(dataset, np.linalg.inv(eigenvectors).T)

########################################################################################################################
