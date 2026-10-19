# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import typing

import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from . import get_bounding_box, get_full_sky, catalog_to_number_density, _build_colorbar

########################################################################################################################

def _get_norm_cmap_label(values: np.ndarray, v_min: typing.Optional[float], v_max: typing.Optional[float], n_sigma: typing.Optional[float], cmap: str, colorbar_label: str, log_scale: bool, assume_positive: bool) -> typing.Tuple[colors.Normalize, colors.Colormap, str, float, float]:

    ####################################################################################################################

    cmap = plt.get_cmap(cmap)

    cmap.set_bad(color = '#808080')

    ####################################################################################################################

    values = values[np.isfinite(values)]

    if values.size == 0:

        raise ValueError('No finite values to display')

    ####################################################################################################################

    if log_scale:

        ################################################################################################################
        # LOG SCALE                                                                                                    #
        ################################################################################################################

        values = values[values > 0.0]

        if values.size == 0:

            raise ValueError('Log scale requires at least one strictly positive value')

        ################################################################################################################

        if v_min is None:
            v_min = np.min(values)

        if v_max is None:
            v_max = np.max(values)

        ################################################################################################################

        if v_min <= 0.0 or v_max <= 0.0:

            raise ValueError('Log scale requires strictly positive limits')

        if v_min > v_max:

            raise ValueError('Invalid value range')

        ################################################################################################################

        return colors.LogNorm(vmin = v_min, vmax = v_max, clip = True), cmap, colorbar_label, v_min, v_max

        ################################################################################################################

    else:

        ################################################################################################################
        # LINEAR SCALE                                                                                                 #
        ################################################################################################################

        if assume_positive and np.max(values) <= 0.0:

            ############################################################################################################

            if v_min is None:
                v_min = 0.0

            if v_max is None:
                v_max = 0.0

        ################################################################################################################

        elif n_sigma is not None and n_sigma > 0.0:

            ############################################################################################################

            v_mean = np.mean(values)
            v_std = np.std(values)

            ############################################################################################################

            if v_min is None:

                v_min = v_mean - n_sigma * v_std

                if not assume_positive or v_min >= 0.0:
                    colorbar_label = 'µ - {}σ < {}'.format(n_sigma, colorbar_label)
                else:
                    v_min = 0.0

            ############################################################################################################

            if v_max is None:

                v_max = v_mean + n_sigma * v_std

                if not assume_positive or v_max >= 0.0:
                    colorbar_label = '{} < µ + {}σ'.format(colorbar_label, n_sigma)
                else:
                    v_max = 0.0

        ################################################################################################################

        if v_min is None:
            v_min = np.min(values)

        if v_max is None:
            v_max = np.max(values)

        ################################################################################################################

        if v_min > v_max:

            raise ValueError('Invalid value range')

        ################################################################################################################

        return colors.Normalize(vmin = v_min, vmax = v_max), cmap, colorbar_label, v_min, v_max

########################################################################################################################

# noinspection PyUnresolvedReferences
def _display(nside: int, footprint: np.ndarray, full_sky: np.ndarray, nest: bool, cmap: str, v_min: float, v_max: float, n_sigma: float, n_hist_bins: int, colorbar_label: str, log_scale: bool, show_colorbar: bool, show_graticule: bool, show_histogram: bool, assume_positive: bool) -> typing.Tuple[plt.Figure, plt.Axes, float, float]:

    ####################################################################################################################

    norm, cmap, label, v_min, v_max = _get_norm_cmap_label(
        full_sky[footprint],
        v_min,
        v_max,
        n_sigma,
        cmap,
        colorbar_label,
        log_scale,
        assume_positive
    )

    ####################################################################################################################

    lon_min, lon_max, lat_min, lat_max = get_bounding_box(nside, footprint, nest)

    ####################################################################################################################
    # HANDLE 0 / 360 DEGREE CROSSING                                                                                   #
    ####################################################################################################################

    if lon_min > lon_max:

        lon_min -= 360.0

    ####################################################################################################################
    # ADD MARGIN AROUND DATA                                                                                           #
    ####################################################################################################################

    x_margin = 0.01
    y_margin = 0.02

    lon_margin = x_margin * (lon_max - lon_min)
    lat_margin = y_margin * (lat_max - lat_min)

    lon_min -= lon_margin
    lon_max += lon_margin

    lat_min = max(-90.0, lat_min - lat_margin)
    lat_max = min(+90.0, lat_max + lat_margin)

    ####################################################################################################################
    # LOCAL SPHERICAL METRIC                                                                                           #
    ####################################################################################################################

    lat_center = 0.5 * (lat_min + lat_max)

    cos_lat = np.cos(np.deg2rad(lat_center))

    if cos_lat <= np.finfo(float).eps:

        raise ValueError('Cartesian projection is singular at the poles')

    ####################################################################################################################
    # PROJECTION SIZE                                                                                                  #
    ####################################################################################################################

    lon_size = (lon_max - lon_min) * cos_lat
    lat_size = (lat_max - lat_min) * 1.00000

    if lon_size <= 0.0 or lat_size <= 0.0:

        raise ValueError('Invalid bounding box')

    ####################################################################################################################

    if lon_size >= lat_size:

        xsize = 1600
        ysize = max(2, int(np.round(xsize * lat_size / lon_size)))

    else:

        ysize = 1600
        xsize = max(2, int(np.round(ysize * lon_size / lat_size)))

    ####################################################################################################################
    # HEALPIX PROJECTION                                                                                               #
    ####################################################################################################################

    projector = hp.projector.CartesianProj(
        lonra = [lon_min, lon_max],
        latra = [lat_min, lat_max],
        xsize = xsize,
        ysize = ysize
    )

    image = projector.projmap(full_sky, lambda x, y, z: hp.vec2pix(nside, x, y, z, nest = nest))

    ####################################################################################################################
    # DISPLAY                                                                                                          #
    ####################################################################################################################

    fig, ax = plt.subplots(figsize = (8, 8))

    img = ax.imshow(
        image,
        extent = (lon_max, lon_min, lat_min, lat_max),
        norm = norm,
        cmap = cmap,
        origin = 'lower',
        interpolation = 'nearest'
    )

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')

    ####################################################################################################################
    # CORRECT SPHERICAL ASPECT RATIO                                                                                   #
    ####################################################################################################################

    ax.set_aspect(1.0 / cos_lat, adjustable = 'box')

    ####################################################################################################################

    if show_colorbar:

        bar = _build_colorbar(ax, img, norm, cmap, n_hist_bins = n_hist_bins, show_histogram = show_histogram, position = 'bottom')

        bar.set_label(label)

    ####################################################################################################################

    if show_graticule:

        ax.grid(True, which = 'major', linestyle = '--', linewidth = 0.5, color = 'red')

    ####################################################################################################################

    fig.tight_layout()

    ####################################################################################################################
    # MATCH COLORBAR WIDTH TO PLOT WIDTH                                                                               #
    ####################################################################################################################

    if show_colorbar:

        fig.canvas.draw()

        ax_position = ax.get_position()
        bar_position = bar.ax.get_position()

        bar.ax.set_axes_locator(None)

        bar.ax.set_position([
            ax_position.x0,
            bar_position.y0,
            ax_position.width,
            bar_position.height
        ])

    ####################################################################################################################

    return fig, ax, v_min, v_max

########################################################################################################################

def display_healpix(nside: int, footprint: np.ndarray, weights: np.ndarray, nest: bool = True, cmap: str = 'jet', v_min: float = None, v_max: float = None, n_sigma: typing.Optional[float] = 2.5, n_hist_bins: int = 100, colorbar_label: str = 'value', log_scale: bool = False, show_colorbar: bool = True, show_graticule: bool = False, show_histogram: bool = True, return_minmax: bool = False, assume_positive: bool = False) -> typing.Union[typing.Tuple[plt.Figure, plt.Axes, float, float], typing.Tuple[plt.Figure, plt.Axes]]:

    """
    Displays a HEALPix map.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the region to display.
    weights : np.ndarray
        HEALPix weights of the region to display.
    nest : bool, default: **True**
        If **True**, ordering scheme is *NESTED*, otherwise, *RING*.
    cmap : str, default: **'jet'**
        Color map.
    v_min : float, default: **None** ≡ :math:`\\mu-n_\\sigma\\cdot\\sigma`
        Minimum range value.
    v_max : float, default: **None** ≡ :math:`\\mu+n_\\sigma\\cdot\\sigma`
        Maximum range value.
    n_sigma : float, default: **2.5**
        Multiplier for standard deviation.
    n_hist_bins : int, default: **100**
        Number of histogram bins in the colorbar.
    colorbar_label : str, default **'value'**
        Colorbar label.
    log_scale : bool, default: **False**
        Specifies whether to enable the logarithm scaling.
    show_colorbar : bool, default: **True**
        Specifies whether to display the colorbar.
    show_graticule : bool, default: **False**
        Specifies whether to display the graticule.
    show_histogram : bool, default: **True**
        Specifies whether to display the colorbar histogram.
    return_minmax : bool, default: **False**
        Specifies whether to return the minimum and maximum values.
    assume_positive : bool, default: **False**
        If True, the input arrays are both assumed to be positive or null values.
    """

    ####################################################################################################################

    if footprint.shape != weights.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = get_full_sky(nside, np.nan, dtype = np.float32, use_zarr = True)

    full_sky[footprint] = np.where(weights != hp.UNSEEN, weights, np.nan)

    ####################################################################################################################

    fig, ax, v_min, v_max = _display(
        nside,
        footprint,
        full_sky,
        nest,
        cmap,
        v_min,
        v_max,
        n_sigma,
        n_hist_bins,
        colorbar_label,
        log_scale,
        show_colorbar,
        show_graticule,
        show_histogram,
        assume_positive
    )

    ####################################################################################################################

    del full_sky

    ####################################################################################################################

    if return_minmax:
        return fig, ax, v_min, v_max
    else:
        return fig, ax

########################################################################################################################

def display_catalog(nside: int, footprint: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool = True, cmap: str = 'jet', v_min: float = None, v_max: float = None, n_sigma: typing.Optional[float] = 2.5, n_hist_bins: int = 100, colorbar_label: str = 'number', log_scale: bool = False, show_colorbar: bool = True, show_graticule: bool = False, show_histogram: bool = True, return_minmax: bool = False, assume_positive: bool = True) -> typing.Union[typing.Tuple[plt.Figure, plt.Axes, float, float], typing.Tuple[plt.Figure, plt.Axes]]:

    """
    Displays a catalog.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the region to display.
    lon : np.ndarray
        Array of longitudes.
    lat : np.ndarray
        Array of latitudes.
    nest : bool, default: **True**
        If **True**, ordering scheme is *NESTED*, otherwise, *RING*.
    cmap : str, default: **'jet'**
        Color map.
    v_min : float, default: **None** ≡ :math:`\\mu-n_\\sigma\\cdot\\sigma`
        Minimum range value.
    v_max : float, default: **None** ≡ :math:`\\mu+n_\\sigma\\cdot\\sigma`
        Maximum range value.
    n_sigma : float, default: **2.5**
        Multiplier for standard deviation.
    n_hist_bins : int, default: **100**
        Number of histogram bins in the colorbar.
    colorbar_label : str, default: **'number'**
        Colorbar label.
    log_scale : bool, default: **False**
        Specifies whether to enable the logarithm scaling.
    show_colorbar : bool, default: **True**
        Specifies whether to display the colorbar.
    show_graticule : bool, default: **False**
        Specifies whether to display the graticule.
    show_histogram : bool, default: **True**
        Specifies whether to display the colorbar histogram.
    return_minmax : bool, default: **False**
        Specifies whether to return the minimum and maximum values.
    assume_positive : bool, default: **True**
        If True, the input arrays are both assumed to be positive or null values.
    """

    ####################################################################################################################

    if lon.shape != lat.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = get_full_sky(nside, np.nan, dtype = np.float32, use_zarr = False)

    catalog_to_number_density(nside, footprint, full_sky, lon, lat, nest = nest, lonlat = True)

    ####################################################################################################################

    fig, ax, v_min, v_max = _display(
        nside,
        footprint,
        full_sky,
        nest,
        cmap,
        v_min,
        v_max,
        n_sigma,
        n_hist_bins,
        colorbar_label,
        log_scale,
        show_colorbar,
        show_graticule,
        show_histogram,
        assume_positive
    )

    ####################################################################################################################

    del full_sky

    ####################################################################################################################

    if return_minmax:
        return fig, ax, v_min, v_max
    else:
        return fig, ax

########################################################################################################################
