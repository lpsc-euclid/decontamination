# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1 import make_axes_locatable

########################################################################################################################

def _build_colorbar(ax: plt.Axes, cmap: colors.Colormap, weights: np.ndarray, v_min: float, v_max: float, show_histogram: bool, n_histogram_bins: int) -> None:

    ####################################################################################################################

    weights = weights[~np.isnan(weights)]

    ####################################################################################################################

    mappable = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = v_min, vmax = v_max))

    cax = make_axes_locatable(ax).append_axes('right', '7.5%', pad = 0.05)

    ####################################################################################################################

    mappable.set_array([])

    colorbar = plt.colorbar(mappable = mappable, cax = cax)

    ####################################################################################################################

    if show_histogram:

        hist, bins = np.histogram(weights, bins = n_histogram_bins)

        colorbar.ax.plot(hist.astype(float) / hist.max(), (bins[:-1] + bins[+1:]) / 2.0, linewidth = 0.75, color = 'k')

        colorbar.ax.set_xticks([0.000, 1.000])
        colorbar.ax.set_yticks([v_min, v_max])

########################################################################################################################

def _setup_ticks(ax: plt.Axes, grid_x: int, grid_y: int, hori_spacing: float, vert_spacing: float) -> None:

    ####################################################################################################################

    if max(grid_y, grid_x) < 10:

        y_interval = 1
        x_interval = 1

    else:

        y_interval = max(1, grid_y // 10)
        x_interval = max(1, grid_x // 10)

    ####################################################################################################################

    ax.set_xticks([j * hori_spacing for j in range(0, grid_y + 1, y_interval)])
    ax.set_yticks([i * vert_spacing for i in range(0, grid_x + 1, x_interval)])

    ####################################################################################################################

    ax.set_xticklabels(range(0, grid_y + 1, y_interval))
    ax.set_yticklabels(range(0, grid_x + 1, x_interval))

########################################################################################################################

def _display_latent_space_big(weights: np.ndarray, cmap: str, show_colorbar: bool, show_histogram: bool, n_histogram_bins: int) -> typing.Tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots()

    ####################################################################################################################

    _setup_ticks(ax, weights.shape[0], weights.shape[1], 1.0, 1.0)

    ####################################################################################################################

    v_min, v_max = np.min(weights), np.max(weights)

    cmap = plt.get_cmap(cmap)

    ####################################################################################################################

    ax.imshow(weights, cmap = cmap)

    ####################################################################################################################

    if show_colorbar:

        _build_colorbar(ax, cmap, weights, v_min, v_max, show_histogram, n_histogram_bins)

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def _display_latent_space_square(weights: np.ndarray, cmap: str, show_colorbar: bool, show_histogram: bool, n_histogram_bins: int) -> typing.Tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots()

    ####################################################################################################################

    # Hexagon with a radius of 1

    hori_len = 3.0 / 2.0
    vert_len = np.sqrt(3.0)

    ####################################################################################################################

    _setup_ticks(ax, weights.shape[0], weights.shape[1], hori_len, vert_len)

    ####################################################################################################################

    v_min, v_max = np.nanmin(weights), np.nanmax(weights)

    cmap = plt.get_cmap(cmap)

    ####################################################################################################################

    for j in range(weights.shape[1]):
        y = j * hori_len

        for i in range(weights.shape[0]):
            x = i * vert_len

            ax.add_patch(patches.Rectangle((y, x), hori_len, vert_len, facecolor = cmap((weights[i, j] - v_min) / (v_max - v_min)), edgecolor = 'none', antialiased = False))

    ####################################################################################################################

    ax.invert_yaxis()

    ax.autoscale_view(scalex = False, scaley = False)

    ax.set_xlim(0, weights.shape[1] * hori_len)
    ax.set_ylim(0, weights.shape[0] * vert_len)

    ####################################################################################################################

    if show_colorbar:

        _build_colorbar(ax, cmap, weights, v_min, v_max, show_histogram, n_histogram_bins)

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def _display_latent_space_hexagonal(weights: np.ndarray, cmap: str, show_colorbar: bool, show_histogram: bool, n_histogram_bins: int) -> typing.Tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots()

    ####################################################################################################################

    # Hexagon with a radius of 1

    hori_len = 3.0 / 2.0
    vert_len = np.sqrt(3.0)

    ####################################################################################################################

    _setup_ticks(ax, weights.shape[0], weights.shape[1], hori_len, vert_len)

    ####################################################################################################################

    v_min, v_max = np.nanmin(weights), np.nanmax(weights)

    cmap = plt.get_cmap(cmap)

    ####################################################################################################################

    for j in range(weights.shape[1]):
        y = j * hori_len

        for i in range(weights.shape[0]):
            x = i * vert_len

            if (j & 1) == 1:

                x += 0.5 * vert_len

            ax.add_patch(patches.RegularPolygon((y, x), numVertices = 6, radius = 1.0, orientation = np.pi / 6, facecolor = cmap((weights[i, j] - v_min) / (v_max - v_min)), edgecolor = 'none', antialiased = False))

    ####################################################################################################################

    ax.invert_yaxis()

    ax.autoscale_view(scalex = False, scaley = False)

    ax.set_xlim(-1.0, (weights.shape[1] - 1) * hori_len + 1.0)
    ax.set_ylim(-0.5 * vert_len, weights.shape[0] * vert_len)

    ####################################################################################################################

    if show_colorbar:

        _build_colorbar(ax, cmap, weights, v_min, v_max, show_histogram, n_histogram_bins)

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def display_latent_space(weights: np.ndarray, topology: str = 'hexagonal', cmap: str = 'viridis', show_frame: bool = True, show_colorbar: bool = True, show_histogram: bool = True, n_histogram_bins: int = 100) -> typing.Tuple[plt.Figure, plt.Axes]:

    """
    Parameters
    ----------
    weights : np.ndarray
        Weights of the map.
    topology : str
        Topology of the map, either **'square'** or **'hexagonal'** (default: **'hexagonal'**).
    cmap : str
        Color map (default: **'viridis'**).
    show_frame : bool
        Specifies whether to display the frame (default: **True**).
    show_colorbar : bool
        Specifies whether to display the colorbar (default: **True**).
    show_histogram : bool
        Specifies whether to display the histogram (default: **True**).
    n_histogram_bins : bool
        ???
    """

    ####################################################################################################################

    if max(weights.shape[0], weights.shape[1]) > 150:

        fig, ax = _display_latent_space_big(weights, cmap, show_colorbar, show_histogram, n_histogram_bins)

    else:

        if topology == 'square':

            fig, ax = _display_latent_space_square(weights, cmap, show_colorbar, show_histogram, n_histogram_bins)

        else:

            fig, ax = _display_latent_space_hexagonal(weights, cmap, show_colorbar, show_histogram, n_histogram_bins)

    ####################################################################################################################

    ax.set_frame_on(show_frame)

    ax.set_aspect('equal')

    ####################################################################################################################

    return fig, ax

########################################################################################################################
