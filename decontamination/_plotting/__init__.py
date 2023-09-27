# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1 import make_axes_locatable

########################################################################################################################

def _build_colorbar(ax: plt.Axes, cmap: colors.Colormap, weights: np.ndarray, show_histogram: bool) -> None:

    ####################################################################################################################

    weights = weights[~np.isnan(weights)]

    v_min = weights.min()
    v_max = weights.max()

    ####################################################################################################################

    mappable = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = v_min, vmax = v_max))

    cax = make_axes_locatable(ax).append_axes('right', '10%' if show_histogram else '7.5%', pad = 0.05)

    ####################################################################################################################

    mappable.set_array([])

    colorbar = plt.colorbar(mappable = mappable, cax = cax)

    ####################################################################################################################

    if show_histogram:

        hist, bins = np.histogram(weights, bins = 100)

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

def _display_square(weights: np.ndarray, cmap: str, show_colorbar: bool, show_histogram: bool) -> typing.Tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots()

    ####################################################################################################################

    _setup_ticks(ax, weights.shape[0], weights.shape[1], 1.0, 1.0)

    ####################################################################################################################

    cmap = plt.get_cmap(cmap)

    ####################################################################################################################

    ax.imshow(weights, cmap = cmap)

    ####################################################################################################################

    if show_colorbar:

        _build_colorbar(ax, cmap, weights, show_histogram)

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def _display_hexagonal(weights: np.ndarray, cmap: str, show_colorbar: bool, show_histogram: bool) -> typing.Tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots()

    ####################################################################################################################

    radius = 0.5

    hori_spacing = np.sqrt(4.0) * radius
    vert_spacing = np.sqrt(3.0) * radius

    _setup_ticks(ax, weights.shape[0], weights.shape[1], hori_spacing * 0.75, vert_spacing * 1.00)

    ####################################################################################################################

    cmap = plt.get_cmap(cmap)

    ####################################################################################################################

    v_min = weights.min()
    v_max = weights.max()

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):

            x = j * hori_spacing * 0.75
            y = i * vert_spacing * 1.00

            if (j & 1) == 1:

                y += vert_spacing * 0.5

            ax.add_patch(patches.RegularPolygon((x, y), numVertices = 6, radius = radius, orientation = np.pi / 6, facecolor = cmap((weights[i, j] - v_min / (v_max - v_min))), edgecolor = 'none'))

    ####################################################################################################################

    ax.autoscale_view(scalex = False, scaley = False)

    ax.set_xlim(-hori_spacing * 0.5, +hori_spacing * (weights.shape[1] * 0.75 - 0.25))
    ax.set_ylim(-vert_spacing * 0.5, +vert_spacing * (weights.shape[0] * 1.00 - 0.00))

    ####################################################################################################################

    if show_colorbar:

        _build_colorbar(ax, cmap, weights, show_histogram)

    ####################################################################################################################

    ax.invert_yaxis()

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def display(weights: np.ndarray, topology: str = 'hexagonal', cmap: str = 'viridis', show_frame: bool = True, show_colorbar: bool = True, show_histogram: bool = True) -> typing.Tuple[plt.Figure, plt.Axes]:

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
    """

    ####################################################################################################################

    if topology == 'square' or max(weights.shape[0], weights.shape[1]) > 150:

        fig, ax = _display_square(weights, cmap, show_colorbar, show_histogram)

    else:

        fig, ax = _display_hexagonal(weights, cmap, show_colorbar, show_histogram)

    ####################################################################################################################

    ax.set_frame_on(show_frame)

    ax.set_aspect('equal')

    ####################################################################################################################

    return fig, ax

########################################################################################################################
