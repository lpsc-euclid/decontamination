# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import numpy as np

from scipy.spatial import distance
from scipy.cluster import hierarchy

########################################################################################################################

class Clustering(object):

    """
    Hierarchical clustering.
    """

    ####################################################################################################################

    @staticmethod
    def clusterize(vectors: np.ndarray, n_clusters: int) -> np.ndarray:

        """
        Clusters the input array using the Lance-Williams hierarchical clustering with complete-linkage algorithm.

        Parameters
        ----------
        vectors : np.ndarray
            Input array to be clustered.
        n_clusters : int
            Desired number of clusters.

        Returns
        -------
        Array giving a cluster identifier for each input vector.
        """

        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:

            raise ValueError('vectors must be a 2D numpy array: (n_samples, n_features)')

        if not isinstance(n_clusters, (int, np.integer)) or n_clusters <= 0:

            raise ValueError('n_clusters must be a strictly positive integer')

        ################################################################################################################

        result = np.full(vectors.shape[0], -1, dtype = int)

        ################################################################################################################
        # CHECK INPUT                                                                                                  #
        ################################################################################################################

        not_nan_mask = ~np.any(np.isnan(vectors), axis = -1)

        n_valid = int(np.count_nonzero(not_nan_mask))

        ################################################################################################################

        if n_valid <= 1:

            if n_valid == 1:

                result[not_nan_mask] = 0

            return result

        ################################################################################################################
        # CLAMP N_CLUSTERS                                                                                             #
        ################################################################################################################

        if n_clusters > n_valid:

            n_clusters = n_valid

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

        ################################################################################################################

        for i in range(vectors.shape[0]):

            if not_nan_mask[i]:

                cluster_id = int(cluster_ids[j])

                if cluster_id in cluster_dict:
                    result[i] = cluster_dict[cluster_id] # k
                else:
                    result[i] = cluster_dict[cluster_id] = k

                    k += 1

                j += 1

        ################################################################################################################

        return result

    ####################################################################################################################

    @staticmethod
    def average(vectors: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:

        """
        Performs averaging per cluster.

        Parameters
        ----------
        vectors : np.ndarray
            Input array be averaged.
        cluster_ids : np.ndarray
            Array of cluster identifiers.
        """

        if vectors.shape[0] != cluster_ids.shape[0]:

            raise ValueError('vectors and cluster_ids must have the same length')

        ################################################################################################################

        result = vectors.copy()

        ################################################################################################################

        for cluster_id in np.unique(cluster_ids):

            if cluster_id >= 0:

                mask = cluster_ids == cluster_id

                result[mask] = np.mean(vectors[mask], axis = 0)

        ################################################################################################################

        return result

########################################################################################################################
