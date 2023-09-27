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
import matplotlib.patches as patches
import matplotlib.lines as lines

########################################################################################################################

som = decontamination.SOM_Online(0, 0, 0)

som.load('random_model.hdf5')

########################################################################################################################

clusters = decontamination.Clustering.clusterize(som.get_weights(), 10)

clustered = decontamination.Clustering.average_over_clusters(som.get_weights(), clusters)

fig, ax = decontamination.display(clustered[:, 0].reshape(som.m, som.n), topology ='squdare')

# decontamination.display(som.get_distance_map(), topology= 'square')

def display_contour(ax, clusters, m, n):

    clusters = clusters.reshape(m, n)

    radius = 0.5

    hori_spacing = np.sqrt(4.0) * radius
    vert_spacing = np.sqrt(3.0) * radius

    for j in range(n):

        x = j * hori_spacing * 0.75

        for i in range(m):

            y = i * vert_spacing * 1.00

            if (j & 1) == 1:

                y += vert_spacing * 0.5

            value = clusters[i, j]

            q1 = i + 1 < m
            q2 = j + 1 < n
            q3 = i - 1 >= 0

            if q1 and value != clusters[i + 1, j]:
                ax.add_patch(patches.RegularPolygon((x, y), numVertices = 4, radius = radius / 2, orientation = np.pi / 6, facecolor = 'red', edgecolor = 'none'))
                ax.add_line(lines.Line2D([x + 0.8 * radius / np.sqrt(3), x - 0.8 * radius / np.sqrt(3)], [y + 0.5 * np.sqrt(3.0) * radius, y + 0.5 * np.sqrt(3.0) * radius], lw=1, color='red'))
                # pass

            if q2 and value != clusters[i, j + 1]:
                # ax.add_patch(patches.RegularPolygon((x, y), numVertices = 4, radius = radius / 2, orientation = np.pi / 6, facecolor = 'red', edgecolor = 'none'))
                # ax.add_line(lines.Line2D([x - hori_spacing * 0.75 / np.sqrt(3), x - 0.5 * hori_spacing * 0.75 / np.sqrt(3)], [y, y + 0.5 * vert_spacing * 1.00], lw=1, color='red'))
                pass

            if q3 and q2 and value != clusters[i - 1, j + 1]:
                # ax.add_patch(patches.RegularPolygon((x, y), numVertices = 4, radius = radius / 2, orientation = np.pi / 6, facecolor = 'red', edgecolor = 'none'))
                pass

display_contour(ax, clusters, som.m, som.n)



########################################################################################################################

if __name__ == '__main__':

    plt.tight_layout()

    plt.show()

########################################################################################################################
