# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb
import healpy as hp

########################################################################################################################

JPLL = np.array([
    1, 3, 5, 7,
    0, 2, 4, 6,
    1, 3, 5, 7,
], dtype = np.int32)

########################################################################################################################

def rand_ang(nside: int, pixels: np.ndarray, nest: bool = False, lonlat = False, rng: np.random.Generator = None, dtype: typing.Type[typing.Union[np.float32, np.float64, float]] = np.float64):

    """
    Samples random spherical coordinates from the given HEALPix pixels.

    See: https://iopscience.iop.org/article/10.1086/427976/pdf

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    pixels : np.ndarray
        ???
    nest : bool
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering (default: **True**).
    lonlat : bool
        If **True**, assumes longitude and latitude in degree, otherwise, co-latitude and longitude in radians (default: **True**).
    rng : np.random.Generator
        ???
    dtype : typing.Type[typing.Union[np.float32, np.float64, float]]
        ???
    """

    ####################################################################################################################

    if nest:
        x, y, f = _nest2hpd(nside, pixels)
    else:
        x, y, f = _ring2hpd(nside, pixels)

    ####################################################################################################################

    if rng is None:

        rng = np.random.default_rng()

    u = rng.random(pixels.shape, dtype = dtype)
    v = rng.random(pixels.shape, dtype = dtype)

    ####################################################################################################################

    z, s, ϕ = _hpd2loc(nside, x, y, f, u, v)

    θ = np.arctan2(s, z)

    ####################################################################################################################

    if lonlat:

        lon = 00.0 + np.degrees(ϕ)
        lat = 90.0 - np.degrees(θ)

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

def _ring2hpd(nside: int, pixels: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Convert ring HEALPix pixel indices to x, y, face (HEALPix Discrete) coordinates."""

    pixels = hp.ring2nest(nside, pixels)

    return _nest2hpd(nside, pixels)

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

    tmp1_pole = m[pole]
    tmp2_pole = 1.0 - tmp1_pole ** 2 / 3.0

    ####################################################################################################################

    z[equa] = h[equa] * 2.0 / 3.0
    z[pole] = r[pole] * tmp2_pole

    s[equa] = np.sqrt(1.0 - z[equa] ** 2)
    s[pole] = np.sqrt(1.0 - tmp2_pole ** 2)

    ϕ[equa] = (JPLL[f[equa]] + (x[equa] - y[equa]) / 1.0000000) * np.pi / 4.0
    ϕ[pole] = (JPLL[f[pole]] + (x[pole] - y[pole]) / tmp1_pole) * np.pi / 4.0

    ####################################################################################################################

    return z, s, ϕ

########################################################################################################################
