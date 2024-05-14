#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import numpy as np

import decontamination

import matplotlib.pyplot as plt

########################################################################################################################

TOPOLOGY = 'square'

N_CLUSTERS = 20

########################################################################################################################

som = decontamination.SOM_Online(0, 0, 0)

som.load('random_model.hdf5')

som._weights[5: 10] = np.nan

som._weights[30: 40] = np.nan

########################################################################################################################

cluster_ids = decontamination.Clustering.clusterize(som.get_weights(), N_CLUSTERS)

clustered_weights = decontamination.Clustering.average(som.get_weights(), cluster_ids)

fig, ax = decontamination.display_latent_space(clustered_weights[:, 0].reshape(som.m, som.n), topology = TOPOLOGY, n_hist_bins = 2 * N_CLUSTERS, cluster_ids = cluster_ids, show_cluster_labels = True)

########################################################################################################################

if __name__ == '__main__':

    plt.tight_layout()

    plt.show()

########################################################################################################################
