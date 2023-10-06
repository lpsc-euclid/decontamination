# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

########################################################################################################################

def _find(parents: list, i: int, j: int) -> tuple:

    while (i, j) != parents[i][j]:

        i, j = parents[i][j]

    return i, j

########################################################################################################################

def _union(parents: list, i1: int, j1: int, i2: int, j2: int) -> None:

    root_i1, root_j1 = _find(parents, i1, j1)
    root_i2, root_j2 = _find(parents, i2, j2)

    parents[root_i1][root_j1] = (root_i2, root_j2)

########################################################################################################################

def compute_cluster_centroids(cluster_ids: np.ndarray) -> typing.List[typing.Tuple[int, int, int]]:

    """
    Compute cluster centroids using the Union-Find algorithm.

    Parameters
    ----------
    cluster_ids : np.ndarray
        Array of cluster identifiers
    """

    ####################################################################################################################

    m, n = cluster_ids.shape

    ####################################################################################################################

    parents = [[(i, j) for j in range(n)] for i in range(m)]

    ####################################################################################################################

    for i in range(m - 1):
        for j in range(n - 1):

            if cluster_ids[i, j] == cluster_ids[i + 1, j]:

                _union(parents, i, j, i + 1, j)

            if cluster_ids[i, j] == cluster_ids[i, j + 1]:

                _union(parents, i, j, i, j + 1)

    ####################################################################################################################

    for j in range(n - 1):

        if cluster_ids[m - 1, j] == cluster_ids[m - 1, j + 1]:

            _union(parents, m - 1, j, m - 1, j + 1)

    ####################################################################################################################

    for i in range(m - 1):

        if cluster_ids[i, n - 1] == cluster_ids[i + 1, n - 1]:

            _union(parents, i, n - 1, i + 1, n - 1)

    ####################################################################################################################

    components = {}

    for i in range(m):
        for j in range(n):

            components.setdefault(_find(parents, i, j), []).append((i, j))

    ####################################################################################################################

    result = []

    for cluster, coords in components.items():

        i_coords, j_coords = zip(*coords)

        result.append((
            cluster_ids[cluster[0]][cluster[1]],
            np.mean(i_coords),
            np.mean(j_coords),
        ))

    ####################################################################################################################

    return result

########################################################################################################################
