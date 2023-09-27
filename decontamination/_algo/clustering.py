# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np
import numba as nb

########################################################################################################################

class Clustering(object):

    @staticmethod
    @nb.njit(parallel = True)
    def _init_distances(weights: np.ndarray) -> np.ndarray:

        n_weights = weights.shape[0]

        distances = np.full((n_weights, n_weights), np.inf, dtype = np.float32)

        for i in nb.prange(n_weights):

            row = distances[i]

            for j in range(i):

                row[j] = np.sum((weights[i] - weights[j]) ** 2)

        return distances

    ####################################################################################################################

    @staticmethod
    @nb.njit
    def _update_clusters(dist: np.ndarray, clusters: np.ndarray) -> None:

        index = np.argmin(dist)
        j, i = divmod(index, dist.shape[0])

        dist[i, :i] = np.maximum(dist[i, :i], dist[j, :i])
        dist[i+1:j, i] = np.maximum(dist[i+1:j, i], dist[j, i+1:j])
        dist[j+1:, i] = np.maximum(dist[j+1:, i], dist[j+1:, j])

        dist[j, :i] = np.inf
        dist[j, i:j] = np.inf
        dist[j+1:, j] = np.inf

        clusters[np.where(clusters == clusters[j])[0]] = clusters[i]

    ####################################################################################################################

    @staticmethod
    def clusterize(weights: np.ndarray, n_clusters: int) -> np.ndarray:

        dist = Clustering._init_distances(weights)

        n_weights = weights.shape[0]

        clusters = np.arange(n_weights)

        for _ in range(n_weights - n_clusters):

            Clustering._update_clusters(dist, clusters)

        return clusters

    ####################################################################################################################

    @staticmethod
    def average_over_clusters(weights: np.ndarray, clusters: np.ndarray) -> np.ndarray:

        result = weights.copy()

        for cluster in np.unique(clusters):

            cluster_indices = np.where(clusters == cluster)

            result[cluster_indices] = np.mean(result[cluster_indices], axis = 0)

        return result

########################################################################################################################
