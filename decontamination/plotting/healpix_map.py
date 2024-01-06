# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

import healpy as hp

import matplotlib.pyplot as plt

from . import catalog_to_number_density

########################################################################################################################

def _get_bounding_box(nside: int, footprint: np.ndarray, nest: bool) -> typing.Tuple[float, float, float, float]:

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

def _display(nside: int, footprint: np.ndarray, sky: np.ndarray, nest: bool, cmap: str, norm: typing.Optional[str], v_min: float, v_max: float) -> typing.Tuple[plt.Figure, plt.Axes]:

    ####################################################################################################################

    lon_min, lon_max, lat_min, lat_max = _get_bounding_box(nside, footprint, nest)

    ####################################################################################################################

    hp.cartview(
        sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        min = v_min,
        max = v_max,
        lonra = [lon_min, lon_max],
        latra = [lat_min, lat_max]
    )

    ####################################################################################################################

    return plt.gcf(), plt.gca()

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
    nest : bool
        If **True**, ordering scheme is *NESTED* (default: **True**).
    cmap : str
        Color map (default: **'jet'**).
    norm : typing.Optional[str]
        Color normalization, **'hist'** = histogram equalized color mapping, **'log'** = logarithmic color mapping (default: **None**).
    v_min : float
        Minimum color scale (default: **None** ≡ min(weights)).
    v_max : float
        Maximum color scale (default: **None** ≡ max(weights)).
    """

    ####################################################################################################################

    if pixels.shape != weights.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype = np.float32)

    full_sky[pixels] = weights

    return _display(
        nside,
        pixels,
        full_sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        v_min = v_min,
        v_max = v_max
    )

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
    nest : bool
        If **True**, ordering scheme is *NESTED* (default: **True**).
    cmap : str
        Color map (default: **'jet'**).
    norm : typing.Optional[str]
        Color normalization, **'hist'** = histogram equalized color mapping, **'log'** = logarithmic color mapping (default: **'hist'**).
    v_min : float
        Minimum color scale (default: **None** ≡ (:math:`\\mu-n_\\sigma\\cdot\\sigma`).
    v_max : float
        Maximum color scale (default: **None** ≡ (:math:`\\mu+n_\\sigma\\cdot\\sigma`).
    """

    ####################################################################################################################

    if lon.shape != lat.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    full_sky = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype = np.float32)

    default_v_min, default_v_max = catalog_to_number_density(nside, pixels, full_sky, lon, lat, nest)

    return _display(
        nside,
        pixels,
        full_sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        v_min = v_min or default_v_min,
        v_max = v_max or default_v_max
    )

########################################################################################################################
