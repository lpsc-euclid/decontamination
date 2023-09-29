# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np

import matplotlib.lines as lines
import matplotlib.pyplot as pyplot

########################################################################################################################

# For a hexagon with a radius of 1:

H_LENGTH = 3.0 / 2.0     # 1.500
V_LENGTH = np.sqrt(3.0)  # 1.732

########################################################################################################################

def display_clusters_square(ax: pyplot.Axes, cluster_ids: np.ndarray) -> None:

    ####################################################################################################################

    m, n = cluster_ids.shape

    ####################################################################################################################

    for j in range(n):
        y = j * H_LENGTH

        for i in range(m):
            x = i * V_LENGTH

            ############################################################################################################

            cluster_id = cluster_ids[i, j]

            ############################################################################################################

            if i + 1 < m and cluster_id != cluster_ids[i + 1, j]:

                ax.add_line(lines.Line2D([
                    y,
                    y + H_LENGTH,
                ], [
                    x + V_LENGTH,
                    x + V_LENGTH,
                ], lw = 1, color = 'black', antialiased = True))

            ############################################################################################################

            if j + 1 < n and cluster_id != cluster_ids[i, j + 1]:

                ax.add_line(lines.Line2D([
                    y + H_LENGTH,
                    y + H_LENGTH,
                ], [
                    x,
                    x + V_LENGTH,
                ], lw = 1, color = 'black', antialiased = True))

########################################################################################################################

def display_clusters_hexagonal(ax: pyplot.Axes, cluster_ids: np.ndarray) -> None:

    ####################################################################################################################

    m, n = cluster_ids.shape

    ####################################################################################################################

    for j in range(n):
        y = j * H_LENGTH

        for i in range(m):
            x = i * V_LENGTH

            i2 = i

            if (j & 1) == 1:

                x += 0.5 * V_LENGTH

                i2 += 1

            ############################################################################################################

            cluster_id = cluster_ids[i, j]

            ############################################################################################################

            if i + 1 < m and cluster_id != cluster_ids[i + 1, j + 0]:

                ax.add_line(lines.Line2D([
                    y - 0.5,
                    y + 0.5,
                ], [
                    x + 0.5 * V_LENGTH,
                    x + 0.5 * V_LENGTH,
                ], lw = 1, color = 'black', antialiased = True))

            ############################################################################################################

            if i2 < m:

                ########################################################################################################

                if j - 1 >= 0 and cluster_id != cluster_ids[i2, j - 1]:

                    ax.add_line(lines.Line2D([
                        y - 1.0,
                        y - 0.5,
                    ], [
                        x - 0.0 * V_LENGTH,
                        x + 0.5 * V_LENGTH,
                    ], lw = 1, color = 'black', antialiased = True))

                ########################################################################################################

                if j + 1 < n and cluster_id != cluster_ids[i2, j + 1]:

                    ax.add_line(lines.Line2D([
                        y + 1.0,
                        y + 0.5,
                    ], [
                        x - 0.0 * V_LENGTH,
                        x + 0.5 * V_LENGTH,
                    ], lw = 1, color = 'black', antialiased = True))

########################################################################################################################

def display_clusters(ax: pyplot.Axes, cluster_ids: np.ndarray, topology: str = 'hexagonal') -> None:

    """
    Parameters
    ----------
    ax : pyplot.Axes
        ???
    cluster_ids : np.ndarray
        Array of cluster ids.
    topology : str
        Topology of the map, either **'square'** or **'hexagonal'** (default: **'hexagonal'**).
    """

    if max(cluster_ids.shape[0], cluster_ids.shape[1]) > 200 or topology == 'square':

        display_clusters_square(ax, cluster_ids)

    else:

        display_clusters_hexagonal(ax, cluster_ids)

########################################################################################################################
