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

TOPOLOGY = 'square2'

N_CLUSTERS = 20

########################################################################################################################

som = decontamination.SOM_Online(0, 0, 0)

som.load('random_model.hdf5')

########################################################################################################################

clusters = decontamination.Clustering.clusterize(som.get_weights(), N_CLUSTERS)

clustered_weights = decontamination.Clustering.average(som.get_weights(), clusters)

fig, ax = decontamination.display_latent_space(clustered_weights[:, 0].reshape(som.m, som.n), topology = TOPOLOGY, n_histogram_bins = N_CLUSTERS)

decontamination.display_clusters(ax, clusters.reshape(som.m, som.n), topology = TOPOLOGY)

ax.set_xlabel('n, j, y')
ax.set_ylabel('m, i, x')

########################################################################################################################

if __name__ == '__main__':

    plt.tight_layout()

    plt.show()

########################################################################################################################
