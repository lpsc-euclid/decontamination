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

from . import get_bounding_box, catalog_to_number_density

########################################################################################################################

_pixels: typing.Optional[np.ndarray] = None
_full_sky: typing.Optional[np.ndarray] = None

########################################################################################################################

def _get_full_sky(nside: int, pixels: np.ndarray) -> np.ndarray:

    global _pixels
    global _full_sky

    ####################################################################################################################

    npix = hp.nside2npix(nside)

    ####################################################################################################################

    if not np.array_equal(_pixels, pixels) or _full_sky is None or _full_sky.shape[0] != npix:

        _pixels = pixels

        _full_sky = np.full(npix, np.nan, dtype = np.float32)

    ####################################################################################################################

    return _full_sky

########################################################################################################################

def _get_limits_and_label(values: np.ndarray, v_min: typing.Optional[float], v_max: typing.Optional[float], n_sigma: float, assume_positive: bool = False, label: str = 'value') -> typing.Tuple[float, float, str]:

    ####################################################################################################################

    _max = np.nanmax(values)
    _mean = np.nanmean(values)
    _std = np.nanstd(values)

    ####################################################################################################################

    if not assume_positive or _max >= 0.0:

        ################################################################################################################

        if v_min is None:

            v_min = _mean - n_sigma * _std

            if not assume_positive or v_min >= 0.0:
                label = 'µ - {}σ < {}'.format(n_sigma, label)
            else:
                v_min = 0.0

        ################################################################################################################

        if v_max is None:

            v_max = _mean + n_sigma * _std

            if not assume_positive or v_max >= 0.0:
                label = '{} < µ + {}σ'.format(label, n_sigma)
            else:
                v_max = 0.0

        ################################################################################################################

        return v_min, v_max, label

    else:

        return 0.0, 0.0, label

########################################################################################################################

def _display(nside: int, footprint: np.ndarray, full_sky: np.ndarray, nest: bool, cmap: str, norm: typing.Optional[str], v_min: float, v_max: float, label: str) -> typing.Tuple[plt.Figure, plt.Axes]:

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

    img = ax.imshow(image, extent = (lon_min, lon_max, lat_min, lat_max), origin = 'lower', aspect = 1.0, cmap = cmap, vmin = v_min, vmax = v_max)

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')

    bar = fig.colorbar(img, ax = ax, orientation = 'horizontal', pad = 0.1, fraction = 0.08)

    bar.set_label(label)

    fig.tight_layout()

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def display_healpix(nside: int, pixels: np.ndarray, weights: np.ndarray, nest: bool = True, cmap: str = 'jet', norm: typing.Optional[str] = None, v_min: float = None, v_max: float = None, n_sigma: float = 2.5, assume_positive: bool = False, label: str = 'value') -> typing.Tuple[plt.Figure, plt.Axes]:

    """
    Displays a HEALPix map.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    pixels : np.ndarray
        HEALPix indices of the region to display.
    weights : np.ndarray
        HEALPix weights of the region to display.
    nest : bool, default: **True**
        If **True**, ordering scheme is *NESTED*, otherwise, *RING*.
    cmap : str, default: **'jet'**
        Color map.
    norm : str, default: **None**
        Optional color normalization, **'hist'** = histogram equalized color mapping, **'log'** = logarithmic color mapping.
    v_min : float, default: **None** ≡ :math:`\\mu-n_\\sigma\\cdot\\sigma)
        Minimum range value.
    v_max : float, default: **None** ≡ :math:`\\mu+n_\\sigma\\cdot\\sigma)
        Maximum range value.
    n_sigma : float, default: **2.5**
        Multiplier for standard deviations to set the resulting v_min and v_max bounds.
    assume_positive : bool, default: **False**
        If True, the input arrays are both assumed to be positive or null values.
    label : str, default **'value'**
        Label of the color bar.
    """

    ####################################################################################################################

    if pixels.shape != weights.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = _get_full_sky(nside, pixels)

    ####################################################################################################################

    full_sky[pixels] = np.where(weights != hp.UNSEEN, weights, np.nan)

    ####################################################################################################################

    v_min, v_max, label = _get_limits_and_label(full_sky[pixels], v_min, v_max, n_sigma, assume_positive = assume_positive, label = label)

    ####################################################################################################################

    return _display(
        nside,
        pixels,
        full_sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        v_min = v_min,
        v_max = v_max,
        label = label
    )

########################################################################################################################

def display_catalog(nside: int, pixels: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool = True, cmap: str = 'jet', norm: typing.Optional[str] = None, v_min: float = None, v_max: float = None, n_sigma: float = 2.5, assume_positive: bool = True, label: str = 'number') -> typing.Tuple[plt.Figure, plt.Axes]:

    """
    Displays a catalog.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    pixels : np.ndarray
        HEALPix indices of the region to display.
    lon : np.ndarray
        Array of longitudes.
    lat : np.ndarray
        Array of latitudes.
    nest : bool, default: **True**
        If **True**, ordering scheme is *NESTED*, otherwise, *RING*.
    cmap : str, default: **'jet'**
        Color map.
    norm : str, default: **'hist'**
        Color normalization, **'hist'** = histogram equalized color mapping, **'log'** = logarithmic color mapping.
    v_min : float, default: **None** ≡ :math:`\\mu-n_\\sigma\\cdot\\sigma`
        Minimum range value.
    v_max : float, default: **None** ≡ :math:`\\mu+n_\\sigma\\cdot\\sigma`
        Maximum range value.
    n_sigma : float, default: **2.5**
        Multiplier for standard deviations to set the resulting v_min and v_max bounds.
    assume_positive : bool, default: **True**
        If True, the input arrays are both assumed to be positive or null values.
    label : str, default **'number'**
        Label for the color bar.
    """

    ####################################################################################################################

    if lon.shape != lat.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = _get_full_sky(nside, pixels)

    ####################################################################################################################

    catalog_to_number_density(nside, pixels, full_sky, lon, lat, nest = nest, lonlat = True)

    ####################################################################################################################

    v_min, v_max, label = _get_limits_and_label(full_sky[pixels], v_min, v_max, n_sigma, assume_positive = assume_positive, label = label)

    ####################################################################################################################

    return _display(
        nside,
        pixels,
        full_sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        v_min = v_min,
        v_max = v_max,
        label = label
    )

########################################################################################################################
