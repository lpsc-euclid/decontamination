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

def catalog_to_number_density(nside: int, pixels: np.ndarray, full_sky: np.ndarray, lon: np.ndarray, lat: np.ndarray, nest: bool = True, lonlat: bool = True, n_sigma: float = 2.0) -> typing.Tuple[float, float]:

    """
    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    pixels : np.ndarray
        HEALPix indices of the region to consider.
    full_sky : np.ndarray
        Resulting full-sky number density (size: :math:`12\\cdot\\mathrm{nside}^2`).
    lon : np.ndarray
        Array of longitudes.
    lat : np.ndarray
        Array of latitudes.
    nest : bool, default: **True**
        If **True**, ordering scheme is *NESTED*.
    lonlat : bool, default: **True**
        If **True**, assumes longitude and latitude in degrees, otherwise, co-latitude and longitude in radians.
    n_sigma : float, default: **2**
        Multiplier for standard deviations to set the resulting number density bounds.

    Returns
    -------
    float
        Lower bound for number density (:math:`\\mu-n_\\sigma\\cdot\\sigma)`.
    float
        Upper bound for number density (:math:`\\mu+n_\\sigma\\cdot\\sigma`).
    """

    ####################################################################################################################

    catalog_pixels = hp.ang2pix(nside, lon, lat, nest = nest, lonlat = lonlat)

    ####################################################################################################################

    full_sky[pixels] = 0.0

    np.add.at(full_sky, catalog_pixels, 1.0)

    ####################################################################################################################

    mean = np.mean(full_sky[pixels])
    std = np.std(full_sky[pixels])

    ####################################################################################################################

    return max(0.0, mean - n_sigma * std), mean + n_sigma * std

########################################################################################################################
