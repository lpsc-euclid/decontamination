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
            Flat input array to be clustered.
        n_clusters : int
            Desired number of clusters.

        Returns
        -------
        Array giving a cluster identifier for each input vector.
        """

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

            cluster_indices = np.nonzero(cluster_ids == cluster_id)[0]

            result[cluster_indices] = np.mean(vectors[cluster_indices], axis = 0)

        return result

########################################################################################################################
