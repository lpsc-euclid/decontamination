# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np
import numba as nb
import numba.cuda as cu

from .. import jit, device_array_full

########################################################################################################################

class Clustering(object):

    ####################################################################################################################

    @staticmethod
    def _init_distances(weights: np.ndarray, enable_gpu: bool = True, threads_per_block: int = 32) -> np.ndarray:

        result = device_array_full((weights.shape[0], weights.shape[0]), np.inf, dtype = np.float32)

        _init_distances_kernel[enable_gpu, (threads_per_block, threads_per_block), (result.shape[0], result.shape[1])](result, weights)

        return result.copy_to_host()

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = False)
    def _update_clusters(dist: np.ndarray, clusters: np.ndarray) -> None:

        ################################################################################################################

        index = np.argmin(dist)

        j, i = divmod(index, dist.shape[0])

        ################################################################################################################

        dist[i, : i] = np.maximum(dist[i, : i], dist[j, : i])
        dist[j + 1:, i] = np.maximum(dist[j + 1:, i], dist[j + 1:, j])
        dist[i + 1: j, i] = np.maximum(dist[i + 1: j, i], dist[j, i + 1: j])

        ################################################################################################################

        dist[j, : i] = np.inf
        dist[j, i: j] = np.inf
        dist[j + 1:, j] = np.inf

        ################################################################################################################

        clusters[np.where(clusters == clusters[j])[0]] = clusters[i]

    ####################################################################################################################

    @staticmethod
    def clusterize(weights: np.ndarray, n_clusters: int, enable_gpu: bool = True, threads_per_block: int = 32) -> np.ndarray:

        """
        Parameters
        ----------
        weights : np.ndarray
            ???
        n_clusters : int
            ???
        """

        ################################################################################################################

        distances = Clustering._init_distances(weights, enable_gpu = enable_gpu, threads_per_block = threads_per_block)

        ################################################################################################################

        result = np.arange(weights.shape[0])

        for _ in range(weights.shape[0] - n_clusters):

            Clustering._update_clusters(distances, result)

        ################################################################################################################

        return result

    ####################################################################################################################

    @staticmethod
    def average(weights: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:

        """
        Parameters
        ----------
        weights : np.ndarray
            ???
        cluster_ids : np.ndarray
            ???
        """

        result = np.empty_like(weights)

        for cluster_id in np.unique(cluster_ids):

            cluster_indices = np.where(cluster_ids == cluster_id)[0]

            result[cluster_indices] = np.nanmean(weights[cluster_indices], axis = 0)

        return result

########################################################################################################################

@jit(kernel = True, parallel = True)
def _init_distances_kernel(result: np.ndarray, weights: np.ndarray) -> None:

    ####################################################################################################################
    # !--BEGIN-CPU--

    for i in nb.prange(weights.shape[0]):

        row = result[i]

        weight_i = weights[i]

        for j in range(i):

            row[j] = np.sum((weight_i - weights[j]) ** 2)

    # !--END-CPU--
    ####################################################################################################################
    # !--BEGIN-GPU--

    i, j = cu.grid(2)

    if i < weights.shape[0] and j < i:

        weight_i = weights[i]
        weight_j = weights[j]

        dist = 0.0

        for k in range(weight_i.shape[0]):

            dist += (weight_i[k] - weight_j[k]) ** 2

        result[i, j] = dist

    # !--END-GPU--

########################################################################################################################
