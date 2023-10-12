# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

import healpy as hp

########################################################################################################################

def _catalog_to_density(nside: int, footprint: np.ndarray, sky: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool):

    ####################################################################################################################

    pixels = hp.ang2pix(nside, lon, lat, nest = nest, lonlat = True)

    ####################################################################################################################

    sky[footprint] = 0.0

    np.add.at(sky, pixels, 1.0)

########################################################################################################################

def display_cart(nside: int, footprint: np.ndarray, sky: np.ndarray, nest: bool = True, cmap: str = 'jet', norm: typing.Optional[str] = None, v_min: float = None, v_max: float = None, title = ''):

    ####################################################################################################################

    lon, lat = hp.pix2ang(nside, footprint, nest = nest, lonlat = True)

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

    lon_min = (lon_center + np.min(d_lon) + 360) % 360
    lon_max = (lon_center + np.max(d_lon) + 360) % 360

    lat_min = np.min(lat)
    lat_max = np.max(lat)

    ####################################################################################################################

    return hp.cartview(
        sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        min = v_min,
        max = v_max,
        lonra = [lon_min, lon_max],
        latra = [lat_min, lat_max],
        title = title,
        return_projected_map = True
    )

########################################################################################################################

def display_healpix(nside, footprint: np.ndarray, weights: np.ndarray, nest: bool = True, cmap: str = 'jet', norm: typing.Optional[str] = None, v_min: float = None, v_max: float = None, title: str = ''):

    sky = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype = np.float32)

    sky[footprint] = weights

    display_cart(
        nside,
        footprint,
        sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        v_min = v_min,
        v_max = v_max,
        title = title
    )

########################################################################################################################

def display_catalog(nside, footprint: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool = True, cmap: str = 'jet', norm: typing.Optional[str] = None, v_min: float = None, v_max: float = None, title: str = ''):

    sky = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype = np.float32)

    _catalog_to_density(nside, footprint, sky, lon, lat, nest)

    display_cart(
        nside,
        footprint,
        sky,
        nest = nest,
        cmap = cmap,
        norm = norm,
        v_min = v_min,
        v_max = v_max,
        title = title
    )

########################################################################################################################
