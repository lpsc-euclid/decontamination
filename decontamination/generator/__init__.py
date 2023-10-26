# -*- coding: utf-8 -*-
########################################################################################################################

import math
import typing

import numpy as np
import numba as nb
import healpy as hp

########################################################################################################################

def get_cell_size(nside: int) -> float:

    """
    In the cartesian plane, returns the HEALPix diamond side length.

    See: https://iopscience.iop.org/article/10.1086/427976/pdf (page 8).

    .. math::
        \\mathrm{cell\\ size}=\\sqrt{\\underbrace{\\left(\\frac{12}{16}\\cdot2\\pi^2\\right)}_{\\mathrm{white\\ area}}/\\underbrace{\\left(12\\cdot\\mathrm{nside}^2\\right)}_{\\mathrm{number\\ of\\ pixels}}}=\\frac{\\pi}{2\\sqrt{2}\\cdot\\mathrm{nside}}

    .. image:: _html_static/healpix_cartesian_plan.svg
        :alt: HEALPix cartesian plan
        :width: 50%
        :align: center

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    """

    return np.pi / (2.0 * math.sqrt(2.0) * nside)

########################################################################################################################

@nb.njit(fastmath = True)
def thetaphi2xy(θ: np.ndarray, ϕ: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

    """
    Performs HEALPix spherical projection from the sphere (θ, ϕ) to the cartesian plane (x, y).

    See: https://iopscience.iop.org/article/10.1086/427976/pdf (page 8).

    Parameters
    ----------
    θ : np.ndarray
        Polar angle (radians :math:`[0,\\pi]`).

    ϕ : np.ndarray
        Longitude angle (radians :math:`[0,2\\pi]`).

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray]
        Coordinates x, y.
    """

    x = np.zeros_like(ϕ)
    y = np.zeros_like(ϕ)
    z = np.cos(θ)

    pole = np.abs(z) > (2.0 / 3.0)
    equa = np.logical_not(pole)
    south = (z < 0.0) & pole

    ####################################################################################################################
    # EQUATORIAL ZONE                                                                                                  #
    ####################################################################################################################

    x[equa] = ϕ[equa]

    y[equa] = 3.0 * np.pi * z[equa] / 8.0

    ####################################################################################################################
    # POLAR CAPS                                                                                                       #
    ####################################################################################################################

    abs_σ = 2.0 - np.sqrt(3.0 * (1.0 - np.abs(z[pole])))

    ####################################################################################################################

    x[pole] = ϕ[pole] - (abs_σ - 1.0) * (ϕ[pole] % (np.pi / 2.0) - np.pi / 4.0)

    y[pole] = np.pi * abs_σ / 4.0

    y[south] *= -1.0

    ####################################################################################################################

    return x, y

########################################################################################################################

@nb.njit(fastmath = True)
def xy2thetaphi(x: np.ndarray, y: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

    """
    Performs HEALPix spherical projection from the cartesian plane (x, y) to the sphere (θ, ϕ).

    See: https://iopscience.iop.org/article/10.1086/427976/pdf (page 8).

    Parameters
    ----------
    x : np.ndarray
        1st coordinate in the healpix plane.
    y : np.ndarray
        2nd coordinate in the healpix plane.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray]
        Coordinates θ, ϕ.
    """

    ϕ = np.zeros_like(x)
    z = np.zeros_like(y)

    equ = np.abs(y) < np.pi / 4.0
    pole = np.logical_not(equ)

    ####################################################################################################################
    # EQUATORIAL ZONE                                                                                                  #
    ####################################################################################################################

    ϕ[equ] = x[equ]

    z[equ] = 8.0 * y[equ] / (3.0 * np.pi)

    ####################################################################################################################
    # POLAR CAPS                                                                                                       #
    ####################################################################################################################

    a = np.abs(y[pole]) - np.pi / 4.0
    b = np.abs(y[pole]) - np.pi / 2.0

    mask = b != 0.0
    ab = np.zeros_like(a)
    ab[mask] = a[mask] / b[mask]
    if not np.all(np.isfinite(ab)):
        raise ValueError('Unexpected projection error')

    ϕ[pole] = x[pole] - ab * (x[pole] % (np.pi / 2.0) - (np.pi / 4.0))

    z[pole] = (1.0 - 1.0 / 3.0 * (2.0 - 4.0 * np.abs(y[pole]) / np.pi) ** 2) * np.sign(y[pole])

    ####################################################################################################################

    return np.arccos(z), ϕ

########################################################################################################################

def rand_ang(nside: int, pixels: np.ndarray, nest: bool = False, lonlat = False, rng: np.random.Generator = None, dtype: typing.Type[typing.Union[np.float32, np.float64, float]] = np.float64):

    """
    Sample random spherical coordinates from the given HEALPix pixels.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    pixels : np.ndarray
        ???
    nest : bool
        ???
    lonlat : bool
        ???
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

JPLL = np.array([
    1, 3, 5, 7,
    0, 2, 4, 6,
    1, 3, 5, 7,
], dtype = np.int32)

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
