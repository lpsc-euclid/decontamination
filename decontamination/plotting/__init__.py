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

def get_limits_and_label(values: np.ndarray, v_min: typing.Optional[float], v_max: typing.Optional[float], n_sigma: float, assume_positive: bool = False, label: str = 'value') -> typing.Tuple[float, float, str]:

    """
    Parameters
    ----------
    values : np.ndarray
        ???
    v_min : float, default: **None** ≡ :math:`\\mu-n_\\sigma\\cdot\\sigma`
        Minimum range value.
    v_max : float, default: **None** ≡ :math:`\\mu+n_\\sigma\\cdot\\sigma`
        Maximum range value.
    n_sigma : float, default: **2.5**
        Multiplier for standard deviations to set the resulting v_min and v_max bounds.
    assume_positive : bool, default: **False**
        If True, the input arrays are both assumed to be positive or null values.
    label : str, default: **'value'**
        Label of the color bar.

    Returns
    -------
    float
        Lower bound for number density (:math:`\\mu-n_\\sigma\\cdot\\sigma)`.
    float
        Upper bound for number density (:math:`\\mu+n_\\sigma\\cdot\\sigma`).
    str
        Updated label of the color bar.
    """

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
