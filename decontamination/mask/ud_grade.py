# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import math
import typing

import numpy as np
import numba as nb

from ..hp import UNSEEN

########################################################################################################################

def ud_grade(nside_in: int, nside_out: int, footprint_in: np.ndarray, footprint_out: np.ndarray, weights: np.ndarray, mode: typing.Optional[str] = None, ignore_zeros: bool = False, log_factor: float = -2.5) -> np.array:

    """
    Upgrades or downgrades the resolutions of HEALPix masks.

    .. warning::
        Note that ud_grade can create artifacts in the power spectra but this implementation tends to minimize them.

    Parameters
    ----------
    nside_in : int
        The HEALPix nside parameter of the input map.
    nside_out : int
        The HEALPix nside parameter of the output map.
    footprint_in : np.ndarray
        HEALPix indices of the input map.
    footprint_out : np.ndarray
        HEALPix indices of the output map.
    weights : np.ndarray
        The input map.
    mode : str, default: **None** ≡ **"arith"**
        Reprojection mode: **"sum"** (galaxy / star number density, ...), **"cov"** (coverage, ...), **"logquad"** (limiting depth, ...), **"log"**, **"quad"** (RMS, PSF, ...), or **"arith"** (galactic extinction, ...) to determine how the input map is rescaled.
    ignore_zeros : bool, default: **False**
        If True, zero values in the input map are ignored during reprojection.
    log_factor : float, default: **-2.5**
        Factor used when processing logarithmic data in the **"logquad"** and **"log"** modes.

    Returns
    -------
    np.ndarray
        The output map.
    """

    if nside_in > nside_out:

        return _downgrade(nside_in, nside_out, footprint_in, footprint_out, weights, mode, ignore_zeros, np.dtype(weights.dtype).type(log_factor))

    if nside_in < nside_out:

        return _upgrade(nside_in, nside_out, footprint_in, footprint_out, weights, mode, ignore_zeros, np.dtype(weights.dtype).type(log_factor))

    return weights

########################################################################################################################

# noinspection DuplicatedCode
@nb.njit
def _downgrade(nside_in: int, nside_out: int, footprint_in: np.array, footprint_out: np.array, weights: np.array, mode: typing.Optional[str], ignore_zeros: bool, log_factor: typing.Any) -> np.array:

    ####################################################################################################################

    npix = int(12 * nside_out * nside_out)

    factor = nside_in // nside_out

    ####################################################################################################################

    sums = np.zeros(npix, dtype = weights.dtype)
    counts = np.zeros(npix, dtype = weights.dtype)

    ####################################################################################################################

    if mode == 'sum':

        ################################################################################################################
        # MODE SUM                                                                                                     #
        ################################################################################################################

        for i in range(len(weights)):

            pix_out = int(np.floor(footprint_in[i] / factor ** 2))
    
            weight = weights[i]

            if not math.isnan(weight) and weight != UNSEEN:
    
                sums[pix_out] += weight
                counts[pix_out] += 1.0000

        map_out = sums * (factor ** 2 / counts)

        ################################################################################################################

    elif mode == 'cov':

        ################################################################################################################
        # MODE COV                                                                                                     #
        ################################################################################################################

        for i in range(len(weights)):

            pix_out = int(np.floor(footprint_in[i] / factor ** 2))

            weight = weights[i]

            if not math.isnan(weight) and weight != UNSEEN:

                sums[pix_out] += weight
                counts[pix_out] += 1.0000

        map_out = sums / counts

        ################################################################################################################

    elif mode == 'logquad':

        ################################################################################################################
        # MODE LOGQUAD                                                                                                 #
        ################################################################################################################

        for i in range(len(weights)):

            pix_out = int(np.floor(footprint_in[i] / factor ** 2))

            weight = weights[i]

            if not math.isnan(weight) and weight != UNSEEN and (not ignore_zeros or weight != 0.0):

                sums[pix_out] += np.power(10.0, weight / log_factor) ** 2
                counts[pix_out] += 1.00000000000000000000000000000000000000

        map_out = log_factor * np.log10(np.sqrt(sums / counts))

        ################################################################################################################

    elif mode == 'log':

        ################################################################################################################
        # MODE LOG                                                                                                     #
        ################################################################################################################

        for i in range(len(weights)):

            pix_out = int(np.floor(footprint_in[i] / factor ** 2))

            weight = weights[i]

            if not math.isnan(weight) and weight != UNSEEN and (not ignore_zeros or weight != 0.0):

                sums[pix_out] += np.power(10.0, weight / log_factor)
                counts[pix_out] += 1.000000000000000000000000000000000

        map_out = log_factor * np.log10(sums / counts)

        ################################################################################################################

    elif mode == 'quad':

        ################################################################################################################
        # MODE QUADRATIC                                                                                               #
        ################################################################################################################

        for i in range(len(weights)):

            pix_out = int(np.floor(footprint_in[i] / factor ** 2))

            weight = weights[i]

            if not math.isnan(weight) and weight != UNSEEN and (not ignore_zeros or weight != 0.0):

                sums[pix_out] += weight ** 2
                counts[pix_out] += 1.000000000

        map_out = np.sqrt(sums / counts)

        ################################################################################################################

    else:

        ################################################################################################################
        # MODE ARITHMETIC                                                                                              #
        ################################################################################################################

        for i in range(len(weights)):

            pix_out = int(np.floor(footprint_in[i] / factor ** 2))

            weight = weights[i]

            if not math.isnan(weight) and weight != UNSEEN and (not ignore_zeros or weight != 0.0):

                sums[pix_out] += weight
                counts[pix_out] += 1.0000

        map_out = sums / counts

        ################################################################################################################

    map_out[counts == 0] = np.nan

    return map_out[footprint_out]

########################################################################################################################

# noinspection PyUnusedLocal
def _upgrade(nside_in: int, nside_out: int, footprint_in: np.array, footprint_out: np.array, weights: np.array, mode: typing.Optional[str], ignore_zeros: bool, log_factor: typing.Any) -> np.array:

    raise ValueError('Upgrading not implemented yet!')

########################################################################################################################
