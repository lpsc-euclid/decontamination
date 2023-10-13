# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

import healpy as hp

import matplotlib.pyplot as plt

########################################################################################################################

def _catalog_to_density(nside: int, footprint: np.ndarray, sky: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool) -> None:

    ####################################################################################################################

    pixels = hp.ang2pix(nside, lon, lat, nest = nest, lonlat = True)

    ####################################################################################################################

    sky[footprint] = 0.0

    np.add.at(sky, pixels, 1.0)

########################################################################################################################

def _get_bounding_box(nside: int, footprint: np.ndarray, nest: bool) -> typing.Tuple[float, float, float, float]:

    ####################################################################################################################
    #                                                                                                                  #
    ####################################################################################################################

    lon, lat = hp.pix2ang(nside, footprint, nest, lonlat = True)

    ####################################################################################################################
    #                                                                                                                  #
    ####################################################################################################################

    lon %= 360

    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    x_mean = np.mean(np.cos(lat_rad) * np.cos(lon_rad))
    y_mean = np.mean(np.cos(lat_rad) * np.sin(lon_rad))

    ####################################################################################################################

    lon_center = np.degrees(np.arctan2(y_mean, x_mean)) % 360

    d_lon = (lon - lon_center + 180) % 360 - 180

    ####################################################################################################################

    return (
        (lon_center + np.min(d_lon) + 360) % 360,
        (lon_center + np.max(d_lon) + 360) % 360,
        np.min(lat),
        np.max(lat),
    )

########################################################################################################################

def _display(nside: int, footprint: np.ndarray, sky: np.ndarray, nest: bool = True, cmap: str = 'jet', norm: typing.Optional[str] = None, v_min: float = None, v_max: float = None) -> typing.Tuple[plt.Figure, plt.Axes]:

    ####################################################################################################################
    #                                                                                                                  #
    ####################################################################################################################

    lon_min, lon_max, lat_min, lat_max = _get_bounding_box(nside, footprint, nest)

    ####################################################################################################################
    #                                                                                                                  #
    ####################################################################################################################

    fig, ax = plt.subplots()

    ####################################################################################################################

    hp.cartview(
        sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        min = v_min,
        max = v_max,
        lonra = [lon_min, lon_max],
        latra = [lat_min, lat_max],
        hold = True
    )

    ####################################################################################################################

    x_ticks = np.linspace(lon_min, lon_max, 5)
    y_ticks = np.linspace(lat_min, lat_max, 5)

    ax.set_xticks(np.linspace(0, 1, len(x_ticks)))
    ax.set_yticks(np.linspace(0, 1, len(y_ticks)))

    ax.set_xticklabels([f'{tick:.2f}' for tick in x_ticks])
    ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks])

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ####################################################################################################################

    return fig, ax

########################################################################################################################

def display_catalog(nside: int, pixels: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool = True, cmap: str = 'jet', norm: typing.Optional[str] = 'hist', v_min: float = None, v_max: float = None) -> typing.Tuple[plt.Figure, plt.Axes]:

    """
    Displays a catalog.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    pixels : np.ndarray
        Array of HEALPix pixels.
    lon : np.ndarray
        Array of longitudes.
    lat : np.ndarray
        Array of latitudes.
    nest : bool
        If **True**, ordering scheme is *NESTED* (default: **True**).
    cmap : str
        Color map (default: **'jet'**).
    norm : typing.Optional[str]
        Color normalization, hist = histogram equalized color mapping, log = logarithmic color mapping (default: **'hist'**).
    v_min : float
        Minimum color scale (default: **None**, uses: min(data)).
    v_max : float
        Maximum color scale (default: **None**, uses: max(data)).
    """

    ####################################################################################################################

    if lon.shape != lat.shape:

        raise ValueError('Invalid shapes')

    ####################################################################################################################

    sky = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype = np.float32)

    _catalog_to_density(nside, pixels, sky, lon, lat, nest)

    return _display(
        nside,
        pixels,
        sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        v_min = v_min,
        v_max = v_max
    )

########################################################################################################################
