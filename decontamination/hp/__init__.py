# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb

########################################################################################################################
# UTILITIES                                                                                                            #
########################################################################################################################

@nb.njit
def _lonlat2θϕ(lon: np.ndarray, lat: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

    θ, ϕ = np.radians(lat), np.radians(lon)
    θ *= -1.0
    θ += np.pi / 2.0
    return θ, ϕ

########################################################################################################################

@nb.njit
def _θϕ2lonlat(θ: np.ndarray, ϕ: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

    lon, lat = np.degrees(ϕ), np.degrees(θ)
    lat *= -1.0
    lat += +90.0
    return lon, lat

########################################################################################################################

@nb.njit(nb.int64[:](nb.int64[:]))
def _spread_bits(v: np.ndarray) -> np.ndarray:

    # Spreads the bits of the provided integer array.

    result = v & 0x00000000FFFFFFFF

    result = (result ^ (result << 0x10)) & 0x0000FFFF0000FFFF
    result = (result ^ (result << 0x08)) & 0x00FF00FF00FF00FF
    result = (result ^ (result << 0x04)) & 0x0F0F0F0F0F0F0F0F
    result = (result ^ (result << 0x02)) & 0x3333333333333333
    result = (result ^ (result << 0x01)) & 0x5555555555555555

    return result

########################################################################################################################

@nb.njit(nb.int64[:](nb.int64[:]))
def _compress_bits(v: np.ndarray) -> np.ndarray:

    # Compresses the bits of the provided integer array.

    result = v & 0x5555555555555555

    result = (result ^ (result >> 0x01)) & 0x3333333333333333
    result = (result ^ (result >> 0x02)) & 0x0F0F0F0F0F0F0F0F
    result = (result ^ (result >> 0x04)) & 0x00FF00FF00FF00FF
    result = (result ^ (result >> 0x08)) & 0x0000FFFF0000FFFF
    result = (result ^ (result >> 0x10)) & 0x00000000FFFFFFFF

    return result

########################################################################################################################

@nb.njit(nb.int64[:](nb.int64, nb.int64[:], nb.int64[:], nb.int64[:]))
def xyf2nest(nside: int, x: np.ndarray, y: np.ndarray, f: np.ndarray) -> np.ndarray:

    # Convert x, y, face (HEALPix Discrete) coordinates to nested HEALPix pixel indices.

    return (
        (_spread_bits(x) << 0)
        +
        (_spread_bits(y) << 1)
        +
        (f * nside ** 2)
    )

########################################################################################################################

@nb.njit(nb.types.Tuple((nb.int64[:], nb.int64[:], nb.int64[:]))(nb.int64, nb.int64[:]))
def nest2xyf(nside: int, pixels: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Convert nested HEALPix pixel indices to x, y, face (HEALPix Discrete) coordinates.

    v = pixels & (nside ** 2 - 1)

    return (
        _compress_bits(v >> 0)
        ,
        _compress_bits(v >> 1)
        ,
        pixels // nside ** 2
    )

########################################################################################################################
# FAST ANG2PIX                                                                                                         #
########################################################################################################################

@nb.njit
def ang2pix(nside: int, θ: np.ndarray, ϕ: np.ndarray, lonlat: bool = False) -> np.ndarray:

    """
    Converts spherical coordinates to HEALPix pixel indices. **Nested ordering only.**

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    θ : np.ndarray
        The θ angular coordinates of a point on the sphere.
    ϕ : np.ndarray
        The ϕ angular coordinates of a point on the sphere.
    lonlat : bool, default: **False**
        If **True**, assumes longitude and latitude in degrees, otherwise, in radians.
    """

    ####################################################################################################################

    x = np.empty_like(θ, dtype = np.int64)
    y = np.empty_like(θ, dtype = np.int64)
    f = np.empty_like(θ, dtype = np.int64)

    ####################################################################################################################

    order = int(np.log2(nside))

    ####################################################################################################################

    if lonlat:

        θ, ϕ = _lonlat2θϕ(θ, ϕ)

    ####################################################################################################################

    z = np.cos(θ)
    za = np.abs(z)

    ϕ = _modulo(ϕ, 2.0 * np.pi) * (2.0 / np.pi)

    mask = (za <= 2.0 / 3.0)
    equa = np.where(mask)[0]
    pole = np.where(~mask)[0]

    ####################################################################################################################
    # EQUA                                                                                                             #
    ####################################################################################################################

    z_equa = z[equa]
    ϕ_equa = ϕ[equa]

    ####################################################################################################################

    temp1 = nside * (0.5 + ϕ_equa)
    temp2 = nside * (z_equa * 0.75)

    jp = (temp1 - temp2).astype(np.int64)
    jm = (temp1 + temp2).astype(np.int64)

    ifp = jp >> order
    ifm = jm >> order

    ####################################################################################################################

    x[equa] = 0x000 + (jm & (nside - 1)) + 0
    y[equa] = nside - (jp & (nside - 1)) - 1

    f[equa] = np.where(
        ifp == ifm,
        np.where(ifp != 4, ifp + 4, 4),
        np.where(ifp < ifm, ifp + 0, ifm + 8)
    )

    ####################################################################################################################
    # POLE                                                                                                             #
    ####################################################################################################################

    z_pole = z[pole]
    ϕ_pole = ϕ[pole]

    ####################################################################################################################

    nϕ = ϕ_pole.astype(np.int64)

    nϕ[nϕ >= 4] = 3

    tp = ϕ_pole - nϕ

    temp = nside * np.sqrt(3.0 * (1.0 - za[pole]))

    jp = ((0.0 + tp) * temp).astype(np.int64)
    jm = ((1.0 - tp) * temp).astype(np.int64)

    jp[jp >= nside] = nside - 1
    jm[jm >= nside] = nside - 1

    ####################################################################################################################

    mask = z_pole >= 0

    x[pole] = np.where(mask, nside - jm - 1, jp)
    y[pole] = np.where(mask, nside - jp - 1, jm)
    f[pole] = np.where(mask, nϕ + 0, nϕ + 8)

    ####################################################################################################################
    # XYF TO NEST                                                                                                      #
    ####################################################################################################################

    return xyf2nest(nside, x, y, f)

########################################################################################################################

@nb.vectorize([nb.float64(nb.float64, nb.float64)])
def _modulo(v1, v2):

    if v1 >= 0.0:
        return v1 if v1 < v2 else v1 % v2
    else:
        return v1 % v2 + v2

########################################################################################################################
# FAST RAND_ANG                                                                                                        #
########################################################################################################################

def rand_ang(nside: int, pixels: np.ndarray, lonlat: bool = False, compat: bool = False, rng: typing.Optional[np.random.Generator] = None):

    """
    Samples random spherical coordinates from the given HEALPix pixels. **Nested ordering only.**

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    pixels : np.ndarray
        HEALPix indices of the region where coordinates are generated.
    lonlat : bool, default: **False**
        If **True**, assumes longitude and latitude in degrees, otherwise, in radians.
    compat : bool, default: **False**
        If **True**, assumes to be compatible with `healpix.randang`.
    rng : np.random.Generator, default: **None** ≡ the default RNG
        Optional random number generator.
    """

    if rng is None:

        rng = np.random.default_rng()

    ####################################################################################################################

    x, y, f = nest2xyf(nside, pixels.astype(np.int64))

    ####################################################################################################################

    if compat:
        u, v = rng.random(pixels.shape[0], dtype = np.float64), rng.random(pixels.shape[0], dtype = np.float64)
    else:
        u, v = rng.random((pixels.shape[0], 2), dtype = np.float64).T

    ####################################################################################################################

    z, s, ϕ = _xyf2loc(nside, x, y, f, u, v)

    θ = np.arctan2(s, z)

    ####################################################################################################################

    return _θϕ2lonlat(θ, ϕ) if lonlat else (θ, ϕ)

########################################################################################################################

# For each face, coordinate of the lowest corner.

LOWEST_CORNER_COORDINATES = np.array([
    1, 3, 5, 7,
    0, 2, 4, 6,
    1, 3, 5, 7,
], dtype = np.int64)

########################################################################################################################

@nb.njit(nb.types.Tuple((nb.float64[:], nb.float64[:], nb.float64[:]))(nb.int64, nb.int64[:], nb.int64[:], nb.int64[:], nb.float64[:], nb.float64[:]))
def _xyf2loc(nside: int, x: np.ndarray, y: np.ndarray, f: np.ndarray, u: np.ndarray, v: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Convert x, y, face (HEALPix Discrete) coordinates to local cylindrical coordinates.

    ####################################################################################################################

    z = np.empty_like(x, dtype = np.float64)
    s = np.empty_like(x, dtype = np.float64)
    ϕ = np.empty_like(x, dtype = np.float64)

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
