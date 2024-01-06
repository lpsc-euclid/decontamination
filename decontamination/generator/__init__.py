# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb

########################################################################################################################

# For each face, coordinate of the lowest corner.

LOWEST_CORNER_COORDINATES = np.array([
    1, 3, 5, 7,
    0, 2, 4, 6,
    1, 3, 5, 7,
], dtype = np.int32)

########################################################################################################################

def healpix_rand_ang(nside: int, pixels: np.ndarray, lonlat = False, rng: typing.Optional[np.random.Generator] = None, dtype: typing.Type[typing.Union[np.float32, np.float64, float]] = np.float64):

    """
    Samples random spherical coordinates from the given HEALPix pixels. Nested ordering only. See:

    * | *HEALPix*: a Framework for High Resolution Discretization, and Fast Analysis of Data Distributed on the Sphere
      | Górski K. M. et al.
      | The Astrophysical Journal, vol. 622 (2005)
        (`iopscience <https://iopscience.iop.org/article/10.1086/427976>`_)

    .. image:: _html_static/healpix_cartesian_plan.svg
        :alt: HEALPix Cartesian Plan
        :width: 50%
        :align: center

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    pixels : np.ndarray
        HEALPix indices of the region where coordinates are generated.
    lonlat : bool
        If **True**, assumes longitude and latitude in degrees, otherwise, co-latitude and longitude in radians (default: **True**).
    rng : typing.Optional[np.random.Generator]
        Random number generator (default: **None** ≡ the default RNG).
    dtype : typing.Type[typing.Union[np.float32, np.float64, float]]
        ??? (default: **np.float64**).
    """

    if rng is None:

        rng = np.random.default_rng()

    ####################################################################################################################

    x, y, f = _nest2hpd(nside, pixels)

    ####################################################################################################################

    uv = rng.random((pixels.shape[0], 2), dtype = dtype).T

    z, s, ϕ = _hpd2loc(nside, x, y, f, uv[0], uv[1])

    θ = np.arctan2(s, z)

    ####################################################################################################################

    if lonlat:

        lon = 00.0 + np.rad2deg(ϕ)
        lat = 90.0 - np.rad2deg(θ)

        return lon, lat

    else:

        return θ, ϕ

########################################################################################################################

@nb.njit(nb.int64[:](nb.int64[:]))
def _compress_bits(v: np.ndarray) -> np.ndarray:

    """Compresses the bits of the provided integer array."""

    result = v & 0x5555555555555555

    result = (result ^ (result >> 0x01)) & 0x3333333333333333
    result = (result ^ (result >> 0x02)) & 0x0F0F0F0F0F0F0F0F
    result = (result ^ (result >> 0x04)) & 0x00FF00FF00FF00FF
    result = (result ^ (result >> 0x08)) & 0x0000FFFF0000FFFF
    result = (result ^ (result >> 0x10)) & 0x00000000FFFFFFFF

    return result

########################################################################################################################

@nb.njit
def _nest2hpd(nside: int, pixels: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Convert nested HEALPix pixel indices to x, y, face (HEALPix Discrete) coordinates."""

    v = pixels & (nside ** 2 - 1)

    return (
        _compress_bits(v >> 0),
        _compress_bits(v >> 1),
        pixels // nside ** 2
    )

########################################################################################################################

@nb.njit(fastmath = True)
def _hpd2loc(nside: int, x: np.ndarray, y: np.ndarray, f: np.ndarray, u: np.ndarray, v: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Convert x, y, face (HEALPix Discrete) coordinates to local cylindrical coordinates."""

    ####################################################################################################################

    z = np.empty_like(u)
    s = np.empty_like(u)
    ϕ = np.empty_like(u)

    ####################################################################################################################

    x = (x + u) / nside
    y = (y + v) / nside

    ####################################################################################################################

    r = 1.0 - np.floor_divide(f, 4.0)
    h = r - 1.0 + (x + y)
    m = 2.0 - r * h

    mask = m > 1.0
    equa = np.where(mask)[0]
    pole = np.where(~mask)[0]

    m_pole_ = m[pole]
    n_pole_ = 1.0 - (m_pole_ ** 2) / 3.0

    ####################################################################################################################

    z[equa] = h[equa] * 2.0 / 3.0
    z[pole] = r[pole] * n_pole_

    s[equa] = np.sqrt(1.0 - z[equa] ** 2)
    s[pole] = np.sqrt(1.0 - n_pole_ ** 2)

    ϕ[equa] = (LOWEST_CORNER_COORDINATES[f[equa]] + (x[equa] - y[equa]) / 1.00000) * np.pi / 4.0
    ϕ[pole] = (LOWEST_CORNER_COORDINATES[f[pole]] + (x[pole] - y[pole]) / m_pole_) * np.pi / 4.0

    ####################################################################################################################

    return z, s, ϕ

########################################################################################################################
