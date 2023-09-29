#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import decontamination

import matplotlib.pyplot as plt

########################################################################################################################

TOPOLOGY = 'square2'

N_CLUSTERS = 20

########################################################################################################################

som = decontamination.SOM_Online(0, 0, 0)

som.load('random_model.hdf5')

########################################################################################################################

cluster_ids = decontamination.Clustering.clusterize(som.get_weights(), N_CLUSTERS)

clustered_weights = decontamination.Clustering.average(som.get_weights(), cluster_ids)

fig, ax = decontamination.display_latent_space(clustered_weights[:, 0].reshape(som.m, som.n), topology = TOPOLOGY, n_histogram_bins = N_CLUSTERS, cluster_ids = cluster_ids)

########################################################################################################################

if __name__ == '__main__':

    plt.tight_layout()

    plt.show()

########################################################################################################################
