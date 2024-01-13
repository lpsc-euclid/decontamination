# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

import healpy as hp

import matplotlib.pyplot as plt

from . import catalog_to_number_density

########################################################################################################################

def get_bounding_box(nside: int, footprint: np.ndarray, nest: bool) -> typing.Tuple[float, float, float, float]:

    ####################################################################################################################
    # PIXELS TO ANGLES                                                                                                 #
    ####################################################################################################################

    lon, lat = hp.pix2ang(nside, footprint, nest, lonlat = True)

    ####################################################################################################################
    # COMPUTE BOUNDING BOX                                                                                             #
    ####################################################################################################################

    lon %= 360

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)

    x_mean = np.mean(np.cos(lat_rad) * np.cos(lon_rad))
    y_mean = np.mean(np.cos(lat_rad) * np.sin(lon_rad))

    ####################################################################################################################

    lon_center = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360

    d_lon = (lon - lon_center + 180) % 360 - 180

    ####################################################################################################################

    return (
        (lon_center + np.min(d_lon) + 360) % 360,
        (lon_center + np.max(d_lon) + 360) % 360,
        np.min(lat),
        np.max(lat),
    )

########################################################################################################################

def _display(nside: int, footprint: np.ndarray, sky: np.ndarray, nest: bool, cmap: str, norm: typing.Optional[str], v_min: float, v_max: float, label: str) -> typing.Tuple[plt.Figure, plt.Axes]:

    ####################################################################################################################

    cmap = plt.get_cmap(cmap)

    cmap.set_bad(color = '#808080')

    ####################################################################################################################

    sky[footprint][sky[footprint] == hp.UNSEEN] = np.nan

    ####################################################################################################################

    lon_min, lon_max, lat_min, lat_max = get_bounding_box(nside, footprint, nest)

    ####################################################################################################################

    projector = hp.projector.CartesianProj(lonra = [lon_min, lon_max], latra = [lat_min, lat_max])

    image = projector.projmap(sky, lambda x, y, z: hp.vec2pix(nside, x, y, z, nest = nest))

    ####################################################################################################################

    fig, ax = plt.subplots(figsize = (10, 7))

    img = ax.imshow(image, extent = (lon_min, lon_max, lat_min, lat_max), origin = 'lower', cmap = cmap, vmin = v_min, vmax = v_max, interpolation = 'none')

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')

    bar = fig.colorbar(img, ax = ax)

    bar.set_label(label)

    ####################################################################################################################

    fig.tight_layout()

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def display_healpix(nside: int, pixels: np.ndarray, weights: np.ndarray, nest: bool = True, cmap: str = 'jet', norm: typing.Optional[str] = None, v_min: float = None, v_max: float = None) -> typing.Tuple[plt.Figure, plt.Axes]:

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
        If **True**, ordering scheme is *NESTED*.
    cmap : str, default: **'jet'**
        Color map.
    norm : typing.Optional[str], default: **None**
        Color normalization, **'hist'** = histogram equalized color mapping, **'log'** = logarithmic color mapping.
    v_min : float, default: **None** ≡ min(weights)
        Minimum color scale.
    v_max : float, default: **None** ≡ max(weights)
        Maximum color scale.
    """

    ####################################################################################################################

    if pixels.shape != weights.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype = np.float32)

    full_sky[pixels] = weights

    ####################################################################################################################

    fig, ax = _display(
        nside,
        pixels,
        full_sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        v_min = v_min,
        v_max = v_max,
        label = 'value'
    )

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def display_catalog(nside: int, pixels: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool = True, cmap: str = 'jet', norm: typing.Optional[str] = None, v_min: float = None, v_max: float = None) -> typing.Tuple[plt.Figure, plt.Axes]:

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
        If **True**, ordering scheme is *NESTED*.
    cmap : str, default: **'jet'**
        Color map.
    norm : typing.Optional[str], default: **'hist'**
        Color normalization, **'hist'** = histogram equalized color mapping, **'log'** = logarithmic color mapping.
    v_min : float, default: **None** ≡ :math:`\\mu-n_\\sigma\\cdot\\sigma`
        Minimum color scale.
    v_max : float, default: **None** ≡ :math:`\\mu+n_\\sigma\\cdot\\sigma`
        Maximum color scale.
    """

    ####################################################################################################################

    if lon.shape != lat.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype = np.float32)

    default_v_min, default_v_max = catalog_to_number_density(nside, pixels, full_sky, lon, lat, nest)

    ####################################################################################################################

    fig, ax = _display(
        nside,
        pixels,
        full_sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        v_min = v_min or default_v_min,
        v_max = v_max or default_v_max,
        label = 'Number of galaxies'
    )

    ####################################################################################################################

    return fig, ax

########################################################################################################################
