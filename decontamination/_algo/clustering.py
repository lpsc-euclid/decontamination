# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np
import numba as nb

########################################################################################################################

class Clustering(object):

    ####################################################################################################################

    @staticmethod
    @nb.njit(parallel = True)
    def _init_distances(weights: np.ndarray) -> np.ndarray:

        ################################################################################################################

        result = np.full((weights.shape[0], weights.shape[0]), np.inf, dtype = np.float32)

        ################################################################################################################

        for i in nb.prange(weights.shape[0]):

            row = result[i]

            weight_i = weights[i]

            for j in range(i):

                row[j] = np.sum((weight_i - weights[j]) ** 2)

        ################################################################################################################

        return result

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
    def clusterize(weights: np.ndarray, n_clusters: int) -> np.ndarray:

        """
        Parameters
        ----------
        weights : np.ndarray
            ???
        n_clusters : int
            ???
        """

        ################################################################################################################

        distances = Clustering._init_distances(weights)

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

        result = weights.copy()

        for cluster_id in np.unique(cluster_ids):

            cluster_indices = np.where(cluster_ids == cluster_id)[0]

            result[cluster_indices] = np.mean(result[cluster_indices], axis = 0)

        return result

########################################################################################################################
