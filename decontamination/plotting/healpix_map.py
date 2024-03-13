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

from . import get_bounding_box, catalog_to_number_density, _build_colorbar

########################################################################################################################

_footprint: typing.Optional[np.ndarray] = None
_full_sky: typing.Optional[np.ndarray] = None

########################################################################################################################

def _get_full_sky(nside: int, footprint: np.ndarray) -> np.ndarray:

    global _footprint
    global _full_sky

    ####################################################################################################################

    npix = hp.nside2npix(nside)

    ####################################################################################################################

    if not np.array_equal(_footprint, footprint) or _full_sky is None or _full_sky.shape[0] != npix:

        full_sky = np.full(npix, np.nan, dtype = np.float32)

        _footprint = footprint
        _full_sky = full_sky

    ####################################################################################################################

    return _full_sky

########################################################################################################################

def _get_limits_norm_label(values: np.ndarray, v_min: typing.Optional[float], v_max: typing.Optional[float], n_sigma: float, colorbar_label: str, log_scale: bool, assume_positive: bool) -> typing.Tuple[float, float, colors.Normalize, str]:

    if log_scale:

        ################################################################################################################
        # LOG SCALE                                                                                                    #
        ################################################################################################################

        values = values[values > 0.0]

        ################################################################################################################

        if v_min is None:
            v_min = np.min(values)

        if v_max is None:
            v_max = np.max(values)

        ################################################################################################################

        return v_min, v_max, colors.LogNorm(vmin = v_min, vmax = v_max), f'log({colorbar_label})'

        ################################################################################################################

    else:

        ################################################################################################################
        # LINEAR SCALE                                                                                                 #
        ################################################################################################################

        _max = np.nanmax(values)
        _mean = np.nanmean(values)
        _std = np.nanstd(values)

        ################################################################################################################

        if not assume_positive or _max >= 0.0:

            ############################################################################################################

            if v_min is None:

                v_min = _mean - n_sigma * _std

                if not assume_positive or v_min >= 0.0:
                    colorbar_label = 'µ - {}σ < {}'.format(n_sigma, colorbar_label)
                else:
                    v_min = 0.0

            ############################################################################################################

            if v_max is None:

                v_max = _mean + n_sigma * _std

                if not assume_positive or v_max >= 0.0:
                    colorbar_label = '{} < µ + {}σ'.format(colorbar_label, n_sigma)
                else:
                    v_max = 0.0

            ############################################################################################################

        else:

            v_min = 0.0
            v_max = 0.0

        ################################################################################################################

        return v_min, v_max, colors.Normalize(vmin = v_min, vmax = v_max), colorbar_label

########################################################################################################################

# noinspection PyUnresolvedReferences
def _display(nside: int, footprint: np.ndarray, full_sky: np.ndarray, nest: bool, cmap: str, v_min: float, v_max: float, n_sigma: float, n_hist_bins: int, colorbar_label: str, log_scale: bool, show_colorbar: bool, show_histogram: bool, assume_positive: bool) -> typing.Tuple[plt.Figure, plt.Axes]:

    ####################################################################################################################

    v_min, v_max, norm, label = _get_limits_norm_label(
        full_sky[footprint],
        v_min,
        v_max,
        n_sigma,
        colorbar_label,
        log_scale,
        assume_positive
    )

    ####################################################################################################################

    cmap = plt.get_cmap(cmap)

    cmap.set_bad(color = '#808080')

    ####################################################################################################################

    lon_min, lon_max, lat_min, lat_max = get_bounding_box(nside, footprint, nest)

    ####################################################################################################################

    projector = hp.projector.CartesianProj(
        lonra = [lon_min, lon_max],
        latra = [lat_min, lat_max],
        xsize = 800,
        ysize = 800
    )

    image = projector.projmap(full_sky, lambda x, y, z: hp.vec2pix(nside, x, y, z, nest = nest))

    ####################################################################################################################

    fig, ax = plt.subplots(figsize = (8, 8))

    img = ax.imshow(image, extent = (lon_min, lon_max, lat_min, lat_max), origin = 'lower', aspect = 1.0, cmap = cmap, norm = norm)

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')

    if show_colorbar:

        bar = _build_colorbar(ax, img, v_min, v_max, cmap, norm, n_hist_bins = n_hist_bins, show_histogram = show_histogram, position = 'bottom')

        bar.set_label(label)

    fig.tight_layout()

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def display_healpix(nside: int, footprint: np.ndarray, weights: np.ndarray, nest: bool = True, cmap: str = 'jet', v_min: float = None, v_max: float = None, n_sigma: float = 2.5, n_hist_bins: int = 100, colorbar_label: str = 'number', log_scale: bool = False, show_colorbar: bool = True, show_histogram: bool = True, assume_positive: bool = False) -> typing.Tuple[plt.Figure, plt.Axes]:

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
        Multiplier for standard deviations.
    n_hist_bins : int, default: **100**
        Number of histogram bins in the colorbar.
    colorbar_label : str, default **'number'**
        Colorbar label.
    log_scale : bool, default: **False**
        Specifies whether to enable the logarithm scaling.
    show_colorbar : bool, default: **True**
        Specifies whether to display the colorbar.
    show_histogram : bool, default: **True**
        Specifies whether to display the colorbar histogram.
    assume_positive : bool, default: **False**
        If True, the input arrays are both assumed to be positive or null values.
    """

    ####################################################################################################################

    if footprint.shape != weights.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = _get_full_sky(nside, footprint)

    full_sky[footprint] = np.where(weights != hp.UNSEEN, weights, np.nan)

    ####################################################################################################################

    return _display(
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
        show_histogram,
        assume_positive
    )

########################################################################################################################

def display_catalog(nside: int, footprint: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool = True, cmap: str = 'jet', v_min: float = None, v_max: float = None, n_sigma: float = 2.5, n_hist_bins: int = 100, colorbar_label: str = 'number', log_scale: bool = False, show_colorbar: bool = True, show_histogram: bool = True, assume_positive: bool = True) -> typing.Tuple[plt.Figure, plt.Axes]:

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
        Multiplier for standard deviations.
    n_hist_bins : int, default: **100**
        Number of histogram bins in the colorbar.
    colorbar_label : str, default **'number'**
        Colorbar label.
    log_scale : bool, default: **False**
        Specifies whether to enable the logarithm scaling.
    show_colorbar : bool, default: **True**
        Specifies whether to display the colorbar.
    show_histogram : bool, default: **True**
        Specifies whether to display the colorbar histogram.
    assume_positive : bool, default: **True**
        If True, the input arrays are both assumed to be positive or null values.
    """

    ####################################################################################################################

    if lon.shape != lat.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = _get_full_sky(nside, footprint)

    catalog_to_number_density(nside, footprint, full_sky, lon, lat, nest = nest, lonlat = True)

    ####################################################################################################################

    return _display(
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
        show_histogram,
        assume_positive
    )

########################################################################################################################
