#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import timeit

import decontamination

import numpy as np
import numba as nb

import matplotlib.pyplot as plt

########################################################################################################################

som_next = decontamination.SOM_Online(0, 0, 0)

som_next.load('random_model.hdf5')

# decontamination.display(som_next.get_centroids()[:, :, 0], topology = 'square')

########################################################################################################################

@nb.njit(parallel = True)
def init_distances(weights: np.ndarray) -> np.ndarray:

    n_weights = weights.shape[0]

    distances = np.full((n_weights, n_weights), np.inf, dtype = np.float32)

    for i in nb.prange(n_weights):

        row = distances[i]

        for j in range(i):

            row[j] = np.sum((weights[i] - weights[j]) ** 2)

    return distances

########################################################################################################################

@nb.njit
def update_clusters(dist: np.ndarray, clusters: np.ndarray) -> None:

    index = np.argmin(dist)
    j, i = divmod(index, dist.shape[0])

    dist[i, :i] = np.maximum(dist[i, :i], dist[j, :i])
    dist[i+1:j, i] = np.maximum(dist[i+1:j, i], dist[j, i+1:j])
    dist[j+1:, i] = np.maximum(dist[j+1:, i], dist[j+1:, j])

    dist[j, :i] = np.inf
    dist[j, i:j] = np.inf
    dist[j+1:, j] = np.inf

    clusters[np.where(clusters == clusters[j])[0]] = clusters[i]

########################################################################################################################

def average_over_clusters(weights: np.ndarray, clusters: np.ndarray) -> None:

    for cluster in np.unique(clusters):

        cluster_indices = np.where(clusters == cluster)

        weights[cluster_indices] = np.mean(weights[cluster_indices], axis = 0)

########################################################################################################################

def cluster(weights: np.ndarray, n_clusters: np.ndarray) -> np.ndarray:

    result = weights.copy()

    dist = init_distances(result)

    n_weights = result.shape[0]

    clusters = np.arange(n_weights)

    for _ in range(n_weights - n_clusters):

        update_clusters(dist, clusters)

    average_over_clusters(result, clusters)

    return result

########################################################################################################################

clustered = cluster(som_next.get_weights(), 10)

decontamination.display(clustered[:, 2].reshape(som_next._m, som_next._n))
decontamination.display(clustered[:, 2].reshape(som_next._m, som_next._n), topology = 'square')


########################################################################################################################

if __name__ == '__main__':

    plt.tight_layout()

    plt.show()

########################################################################################################################
