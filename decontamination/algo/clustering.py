# -*- coding: utf-8 -*-
########################################################################################################################

import math
import timeit

import numpy as np
import numba as nb
import numba.cuda as cu

from scipy.spatial import distance
from scipy.cluster import hierarchy

from .. import jit, device_array_full

########################################################################################################################

class Clustering(object):

    """
    Hierarchical clustering.
    """

    ####################################################################################################################

    @staticmethod
    def _init_distances(weights: np.ndarray, enable_gpu: bool, threads_per_blocks: int) -> np.ndarray:

        result = device_array_full(2 * (weights.shape[0], ), np.inf, dtype = np.float32)

        _init_distances_kernel[enable_gpu, 2 * (threads_per_blocks, ), 2 * (weights.shape[0], )](
            result,
            weights
        )

        return result.copy_to_host()

    ####################################################################################################################

    @staticmethod
    @nb.njit()
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
    def _clusterize(vectors: np.ndarray, n_clusters: int, enable_gpu: bool = True, threads_per_blocks: int = 32) -> np.ndarray:

        """
        Clusters the input array using the Lance-Williams hierarchical clustering with complete-linkage algorithm.

        Parameters
        ----------
        vectors : np.ndarray
            Flat input array to be clustered.
        n_clusters : int
            Desired number of clusters.
        enable_gpu : bool
            If available, run on GPU rather than CPU (default: **True**).
        threads_per_blocks : int
            Number of GPU threads per blocks (default: **32**).

        Note
        ----

        Number of iterations: `weights.shape[0] - n_clusters`.

        Return
        ------
        Array giving a cluster identifier for each input vector.
        """

        ################################################################################################################
        # COMPUTE DISTANCES                                                                                            #
        ################################################################################################################

        time1 = timeit.default_timer()
        distances = Clustering._init_distances(vectors, enable_gpu = enable_gpu, threads_per_blocks = threads_per_blocks)
        time2 = timeit.default_timer()

        ################################################################################################################
        # COMPUTE CLUSTERS                                                                                             #
        ################################################################################################################

        result = np.arange(vectors.shape[0])

        nan_mask = np.any(np.isnan(vectors), axis = -1)

        for _ in range(vectors.shape[0] - nan_mask.sum() - n_clusters):

            Clustering._update_clusters(distances, result)

        ################################################################################################################

        result[nan_mask] = -1

        time3 = timeit.default_timer()

        print(time2 - time1, time3 - time2)

        ################################################################################################################
        # REINDEX CLUSTERS                                                                                             #
        ################################################################################################################

        cnt = 0

        cluster_dict = {}

        for i, cluster_id in enumerate(result):

            if not nan_mask[i]:

                if cluster_id in cluster_dict:
                    result[i] = cluster_dict[cluster_id] # cnt
                else:
                    result[i] = cluster_dict[cluster_id] = cnt

                    cnt += 1

        ################################################################################################################

        return result

    ####################################################################################################################

    @staticmethod
    def clusterize(vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Clusters the input array using the Lance-Williams hierarchical clustering with complete-linkage algorithm.

        Parameters
        ----------
        vectors : np.ndarray
            Flat input array to be clustered.
        n_clusters : int
            Desired number of clusters.

        Return
        ------
        Array giving a cluster identifier for each input vector.
        """

        time1 = timeit.default_timer()

        ################################################################################################################

        not_nan_mask = ~np.any(np.isnan(vectors), axis = -1)

        ################################################################################################################
        # COMPUTE DISTANCES                                                                                            #
        ################################################################################################################

        distances = distance.pdist(vectors[not_nan_mask])

        ################################################################################################################
        # COMPUTE CLUSTERS                                                                                             #
        ################################################################################################################

        z = hierarchy.linkage(distances, method = 'complete')

        cluster_ids = hierarchy.fcluster(z, n_clusters, criterion = 'maxclust')

        ################################################################################################################
        # REINDEX CLUSTERS                                                                                             #
        ################################################################################################################

        j = 0
        k = 0

        cluster_dict = {}

        result = np.full(vectors.shape[0], -1, dtype = int)

        ################################################################################################################

        for i in range(vectors.shape[0]):

            if not_nan_mask[i]:

                cluster_id = cluster_ids[j]

                if cluster_id in cluster_dict:
                    result[i] = cluster_dict[cluster_id] # k
                else:
                    result[i] = cluster_dict[cluster_id] = k

                    k += 1

                j += 1

        ################################################################################################################

        time2 = timeit.default_timer()

        print(time2 - time1)

        return result

    ####################################################################################################################

    @staticmethod
    def average(vectors: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:

        """
        Performs averaging per cluster.

        Parameters
        ----------
        vectors : np.ndarray
            Flat input array be averaged.
        cluster_ids : np.ndarray
            Array of cluster identifiers.
        """

        result = np.empty_like(vectors)

        for cluster_id in np.unique(cluster_ids):

            cluster_indices = np.where(cluster_ids == cluster_id)[0]

            result[cluster_indices] = np.mean(vectors[cluster_indices], axis = 0)

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

            weight_j = weights[j]

            dist = np.sum((weight_i - weight_j) ** 2)

            row[j] = np.inf if math.isnan(dist) else dist

    # !--END-CPU--
    ####################################################################################################################
    # !--BEGIN-GPU--

    i, j = cu.grid(2)

    if j < i < weights.shape[0]:

        dist = 0.0

        weight_i = weights[i]
        weight_j = weights[j]

        for k in range(weight_i.shape[0]):

            dist += (weight_i[k] - weight_j[k]) ** 2

        result[i, j] = np.inf if math.isnan(dist) else dist

    # !--END-GPU--

########################################################################################################################