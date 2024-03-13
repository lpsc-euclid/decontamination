# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import typing

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

from . import _build_colorbar, clustering

########################################################################################################################

# For a hexagon with a radius of 1:

H_LENGTH = 3.0 / 2.0     # 1.500
V_LENGTH = np.sqrt(3.0)  # 1.732

########################################################################################################################

def _init_plot(weights: np.ndarray, v_min: float, v_max: float, cmap: str, log_scale: bool) -> typing.Tuple[plt.Figure, plt.Axes, float, float, colors.Normalize, colors.Colormap]:

    ####################################################################################################################
    # INIT FIGURE & AXES                                                                                               #
    ####################################################################################################################

    fig, ax = plt.subplots()

    ####################################################################################################################
    # INIT TICKS                                                                                                       #
    ####################################################################################################################

    if max(weights.shape) < 10:

        y_interval = 0x0000000000000000000000000001
        x_interval = 0x0000000000000000000000000001

    else:

        y_interval = max(1, weights.shape[1] // 10)
        x_interval = max(1, weights.shape[0] // 10)

    ####################################################################################################################

    ax.set_xticks(np.arange(0, (weights.shape[1] + 1) * H_LENGTH, y_interval * H_LENGTH))
    ax.set_yticks(np.arange(0, (weights.shape[0] + 1) * V_LENGTH, x_interval * V_LENGTH))

    ax.set_xticklabels(range(0, (weights.shape[1] + 1) * 0x000001, y_interval * 0x000001))
    ax.set_yticklabels(range(0, (weights.shape[0] + 1) * 0x000001, x_interval * 0x000001))

    ####################################################################################################################
    # INIT MIN & MAX                                                                                                   #
    ####################################################################################################################

    values = weights[weights > 0.0] if log_scale else weights

    ####################################################################################################################

    if v_min is None:
        v_min = np.nanmin(values)

    if v_max is None:
        v_max = np.nanmax(values)

    ####################################################################################################################
    # INIT NORM                                                                                                        #
    ####################################################################################################################

    if log_scale:
        norm = colors.LogNorm(vmin = v_min, vmax = v_max)
    else:
        norm = colors.Normalize(vmin = v_min, vmax = v_max)

    ####################################################################################################################
    # INIT CMAP                                                                                                        #
    ####################################################################################################################

    cmap = plt.get_cmap(cmap)

    cmap.set_bad(color = 'gray')

    ####################################################################################################################

    return fig, ax, v_min, v_max, norm, cmap

########################################################################################################################

def _display_latent_space_big(ax: plt.Axes, weights: np.ndarray, cmap: colors.Colormap, norm: colors.Normalize) -> None:

    ####################################################################################################################

    ax.imshow(weights, cmap = cmap, norm = norm, extent = (
        0, weights.shape[1] * H_LENGTH,
        weights.shape[0] * V_LENGTH, 0,
    ))

    ####################################################################################################################

    ax.set_xlim(0, weights.shape[1] * H_LENGTH)
    ax.set_ylim(0, weights.shape[0] * V_LENGTH)

########################################################################################################################

def _display_latent_space_square(ax: plt.Axes, weights: np.ndarray, cmap: colors.Colormap, norm: colors.Normalize, antialiased: bool) -> None:

    ####################################################################################################################

    for j in range(weights.shape[1]):
        y = j * H_LENGTH

        for i in range(weights.shape[0]):
            x = i * V_LENGTH

            ax.add_patch(patches.Rectangle((y, x), H_LENGTH, V_LENGTH, facecolor = cmap(norm(weights[i, j])), edgecolor = 'none', antialiased = antialiased))

    ####################################################################################################################

    ax.set_xlim(0, weights.shape[1] * H_LENGTH)
    ax.set_ylim(0, weights.shape[0] * V_LENGTH)

########################################################################################################################

def _display_latent_space_hexagonal(ax: plt.Axes, weights: np.ndarray, cmap: colors.Colormap, norm: colors.Normalize, antialiased: bool) -> None:

    ####################################################################################################################

    for j in range(weights.shape[1]):
        y = j * H_LENGTH

        for i in range(weights.shape[0]):
            x = i * V_LENGTH

            if (j & 1) == 1:

                x += 0.5 * V_LENGTH

            ax.add_patch(patches.RegularPolygon((y, x), numVertices = 6, radius = 1.0, orientation = np.pi / 6.0, facecolor = cmap(norm(weights[i, j])), edgecolor = 'none', antialiased = antialiased))

    ####################################################################################################################

    ax.set_xlim(-1.0, (weights.shape[1] - 1) * H_LENGTH + 1.0)
    ax.set_ylim(-0.5 * V_LENGTH, weights.shape[0] * V_LENGTH)

########################################################################################################################

def display_latent_space(weights: np.ndarray, topology: typing.Optional[str] = None, v_min: float = None, v_max: float = None, cmap: str = 'viridis', n_hist_bins: int = 100, cluster_ids: typing.Optional[np.ndarray] = None, log_scale: bool = False, antialiased: bool = False, show_frame: bool = True, show_colorbar: bool = True, show_histogram: bool = True, show_cluster_labels: bool = False) -> typing.Tuple[plt.Figure, plt.Axes]:

    """
    Displays the latent space.

    Parameters
    ----------
    weights : np.ndarray
        Weights of the map.
    topology : str, default: **None** ≡ **'hexagonal'**
        Topology of the map, either **'square'** or **'hexagonal'**.
    v_min : float, default: **None** ≡ min(weights)
        Minimum range value.
    v_max : float, default: **None** ≡ max(weights)
        Maximum range value.
    cmap : str, default: **'viridis'**
        Color map.
    n_hist_bins : int, default: **100**
        Number of histogram bins in the colorbar.
    cluster_ids : np.ndarray, default: **None**
        Optional array of cluster identifiers (see :class:`Clustering <decontamination.algo.clustering.Clustering>`).
    log_scale : bool, default: **False**
        Specifies whether to enable the logarithm scaling.
    antialiased : bool, default: **False**
        Specifies whether to enable the antialiasing.
    show_frame : bool, default: **True**
        Specifies whether to display the frame.
    show_colorbar : bool, default: **True**
        Specifies whether to display the colorbar.
    show_histogram : bool, default: **True**
        Specifies whether to display the colorbar histogram.
    show_cluster_labels : bool, default: **False**
        Specifies whether to display the cluster labels.
    """

    ####################################################################################################################

    if len(weights.shape) != 2:

        raise ValueError('Invalid latent space shape, must be (m, n)')

    ####################################################################################################################

    fig, ax, v_min, v_max, norm, cmap = _init_plot(weights, v_min, v_max, cmap, log_scale)

    ####################################################################################################################

    if max(weights.shape) > 200:

        _display_latent_space_big(ax, weights, cmap, norm)

    else:

        if topology == 'square':

            _display_latent_space_square(ax, weights, cmap, norm, antialiased)

        else:

            _display_latent_space_hexagonal(ax, weights, cmap, norm, antialiased)

    ####################################################################################################################

    if show_colorbar:

        _build_colorbar(ax, weights, cmap, norm, n_hist_bins, show_histogram, position = 'right')

    ####################################################################################################################

    if cluster_ids is not None:

        clustering.display_clusters(ax, cluster_ids.reshape(weights.shape[0], weights.shape[1]), topology = topology, show_cluster_labels = show_cluster_labels)

    ####################################################################################################################

    ax.set_frame_on(show_frame)

    ax.set_aspect('equal')

    ax.invert_yaxis()

    ####################################################################################################################

    return fig, ax

########################################################################################################################
