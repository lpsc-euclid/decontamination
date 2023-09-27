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

som = decontamination.SOM_Online(0, 0, 0)

som.load('random_model.hdf5')

topology = 'square2'

########################################################################################################################

clusters = decontamination.Clustering.clusterize(som.get_weights(), 20)

clustered_weights = decontamination.Clustering.average_over_clusters(som.get_weights(), clusters)

fig, ax = decontamination.display_latent_space(clustered_weights[:, 0].reshape(som.m, som.n), topology = topology, n_histogram_bins = np.unique(clusters).shape[0])

decontamination.display_clusters(ax, clusters.reshape(som.m, som.n), topology = topology)

ax.set_xlabel('n, j, y')
ax.set_ylabel('m, i, x')

########################################################################################################################

if __name__ == '__main__':

    plt.tight_layout()

    plt.show()

########################################################################################################################
