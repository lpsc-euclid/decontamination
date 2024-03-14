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
import matplotlib.image as image
import matplotlib.colors as colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

########################################################################################################################

def get_bounding_box(nside: int, footprint: np.ndarray, nest: bool = True) -> typing.Tuple[float, float, float, float]:

    """
    Get the bounding box of the given footprint.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the region to consider.
    nest : bool, default: **True**
        If **True**, ordering scheme is *NESTED*, otherwise, *RING*.
    """

    ####################################################################################################################
    # PIXELS TO ANGLES                                                                                                 #
    ####################################################################################################################

    lon, lat = hp.pix2ang(nside, footprint, nest, lonlat = True)

    ####################################################################################################################
    # COMPUTE BOUNDING BOX                                                                                             #
    ####################################################################################################################

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    x_mean = np.mean(np.cos(lat_rad) * np.cos(lon_rad))
    y_mean = np.mean(np.cos(lat_rad) * np.sin(lon_rad))

    ####################################################################################################################

    lon_center = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360

    d_lon = (lon % 306 - lon_center + 180) % 360 - 180

    ####################################################################################################################

    return (
        (lon_center + np.min(d_lon) + 360) % 360,
        (lon_center + np.max(d_lon) + 360) % 360,
        np.min(lat),
        np.max(lat),
    )

########################################################################################################################

def catalog_to_number_density(nside: int, footprint: np.ndarray, full_sky: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool = True, lonlat: bool = True) -> None:

    """
    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the region to consider.
    full_sky : np.ndarray
        Resulting full-sky number density (size must be :math:`12\\cdot\\mathrm{nside}^2`).
    lon : np.ndarray
        Array of longitudes.
    lat : np.ndarray
        Array of latitudes.
    nest : bool, default: **True**
        If **True**, ordering scheme is *NESTED*, otherwise, *RING*.
    lonlat : bool, default: **True**
        If **True**, assumes longitude and latitude in degrees, otherwise, co-latitude and longitude in radians.
    """

    ####################################################################################################################

    catalog_pixels = hp.ang2pix(nside, lon, lat, nest = nest, lonlat = lonlat)

    ####################################################################################################################

    full_sky[footprint] = 0.0

    np.add.at(full_sky, catalog_pixels, 1.0)

########################################################################################################################

def _build_colorbar(
    ax: plt.Axes,
    weights: typing.Union[np.ndarray, image.AxesImage],
    norm: colors.Normalize,
    cmap: colors.Colormap,
    n_hist_bins: int = 100,
    show_histogram: bool = True,
    position: str = 'right',
    size: typing.Optional[typing.Union[float, str]] = None,
    pad: typing.Optional[typing.Union[float, str]] = None
) -> plt.colorbar:

    ####################################################################################################################

    if isinstance(weights, image.AxesImage):

        weights = weights.get_array()

    ####################################################################################################################

    if not size:
        size = '8%'

    if position in ['left', 'right']:
        orientation = 'vertical'
        if not pad:
            pad = 0.05
    elif position in ['top', 'bottom']:
        orientation = 'horizontal'
        if not pad:
            pad = 0.55
    else:
        orientation = 'vertical'

    ####################################################################################################################

    ad = make_axes_locatable(ax)

    result = plt.colorbar(
        mappable = plt.cm.ScalarMappable(cmap = cmap, norm = norm),
        cax = ad.append_axes(position, size, pad = pad),
        orientation = orientation,
    )

    ####################################################################################################################

    if show_histogram:

        hist, bins = np.histogram(weights[np.isfinite(weights)], bins = np.logspace(np.log10(norm.vmin), np.log10(norm.vmax), n_hist_bins) if isinstance(norm, colors.LogNorm) else n_hist_bins)

        if position in ['left', 'right']:
            result.ax.plot(hist.astype(float) / hist.max(), bins[: -1], linewidth = 0.75, color = 'k')
        if position in ['top', 'bottom']:
            result.ax.plot(bins[: -1], hist.astype(float) / hist.max(), linewidth = 0.75, color = 'k')

    ####################################################################################################################

    return result

########################################################################################################################
