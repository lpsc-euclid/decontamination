# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np

import matplotlib.lines as lines
import matplotlib.pyplot as pyplot

########################################################################################################################

def display_clusters_square(ax: pyplot.Axes, cluster_ids: np.ndarray) -> None:

    ####################################################################################################################

    # Hexagon with a radius of 1

    hori_len = 3.0 / 2.0
    vert_len = np.sqrt(3.0)

    ####################################################################################################################

    m, n = cluster_ids.shape

    ####################################################################################################################

    for j in range(n):
        y = j * hori_len

        for i in range(m):
            x = i * vert_len

            ############################################################################################################

            cluster_id = cluster_ids[i, j]

            ############################################################################################################

            if i + 1 < m and cluster_id != cluster_ids[i + 1, j]:

                ax.add_line(lines.Line2D([
                    y,
                    y + hori_len,
                ], [
                    x + vert_len,
                    x + vert_len,
                ], lw = 1, color = 'black'))

            ############################################################################################################

            if j + 1 < n and cluster_id != cluster_ids[i, j + 1]:

                ax.add_line(lines.Line2D([
                    y + hori_len,
                    y + hori_len,
                ], [
                    x,
                    x + vert_len,
                ], lw = 1, color = 'black'))

########################################################################################################################

def display_clusters_hexagonal(ax: pyplot.Axes, cluster_ids: np.ndarray) -> None:

    ####################################################################################################################

    # Hexagon with a radius of 1

    hori_len = 3.0 / 2.0
    vert_len = np.sqrt(3.0)

    ####################################################################################################################

    m, n = cluster_ids.shape

    ####################################################################################################################

    for j in range(n):
        y = j * hori_len

        for i in range(m):
            x = i * vert_len

            i2 = i

            if (j & 1) == 1:

                x += 0.5 * vert_len

                i2 += 1

            ############################################################################################################

            cluster_id = cluster_ids[i, j]

            ############################################################################################################

            if i + 1 < m and cluster_id != cluster_ids[i + 1, j + 0]:

                ax.add_line(lines.Line2D([
                    y - 0.5,
                    y + 0.5,
                ], [
                    x + 0.5 * vert_len,
                    x + 0.5 * vert_len,
                ], lw = 1, color = 'black'))

            ############################################################################################################

            if i2 < m:

                ########################################################################################################

                if j - 1 >= 0 and cluster_id != cluster_ids[i2, j - 1]:

                    ax.add_line(lines.Line2D([
                        y - 1.0,
                        y - 0.5,
                    ], [
                        x - 0.0 * vert_len,
                        x + 0.5 * vert_len,
                    ], lw = 1, color = 'black'))

                ########################################################################################################

                if j + 1 < n and cluster_id != cluster_ids[i2, j + 1]:

                    ax.add_line(lines.Line2D([
                        y + 1.0,
                        y + 0.5,
                    ], [
                        x - 0.0 * vert_len,
                        x + 0.5 * vert_len,
                    ], lw = 1, color = 'black'))

########################################################################################################################

def display_clusters(ax: pyplot.Axes, cluster_ids: np.ndarray, topology: str = 'hexagonal') -> None:

    """
    Parameters
    ----------
    ax : pyplot.Axes
        ???
    cluster_ids : np.ndarray
        ???
    topology : str
        Topology of the map, either **'square'** or **'hexagonal'** (default: **'hexagonal'**).
    """

    if max(cluster_ids.shape[0], cluster_ids.shape[1]) > 150:

        raise Exception('Method not implemented for map size > 150')

    else:

        if topology == 'square':

            display_clusters_square(ax, cluster_ids)

        else:

            display_clusters_hexagonal(ax, cluster_ids)

########################################################################################################################
