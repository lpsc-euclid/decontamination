# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb

########################################################################################################################

def get_cell_size(nside: int) -> float:

    """
    In the cartesian plane, returns the side length of the HEALPix diamonds.

    See: https://iopscience.iop.org/article/10.1086/427976/pdf (page 8).

    .. math::
        \\left(\\frac{12}{16}2\\pi^2\\right)/\\left(12\\cdot\\mathrm{nside}^2\\right)

    .. image:: _static/healpix_cartesian_plan.svg
        :alt: HEALPix cartesian plan
        :width: 50%
        :align: center

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    """

    full_sky = (12.0 / 16.0) * (2.0 * np.pi * np.pi)

    n_pixels = 12 * nside * nside

    return full_sky / n_pixels

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

    σ = 2.0 - np.sqrt(3.0 * (1.0 - np.abs(z[pole])))

    ####################################################################################################################

    x[pole] = ϕ[pole] - (np.abs(σ) - 1.0) * (ϕ[pole] % (np.pi / 2.0) - np.pi / 4.0)

    y[pole] = np.pi * σ / 4.0

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
        raise ValueError('unexpected projection error')

    ϕ[pole] = x[pole] - ab * (x[pole] % (np.pi / 2.0) - (np.pi / 4.0))

    z[pole] = (1.0 - 1.0 / 3.0 * (2.0 - 4.0 * np.abs(y[pole]) / np.pi) ** 2) * y[pole] / np.abs(y[pole])

    ####################################################################################################################

    return np.arccos(z), ϕ

########################################################################################################################
