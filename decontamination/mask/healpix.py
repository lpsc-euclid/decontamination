# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import tqdm
import typing

import numpy as np
import numba as nb

from ..hp import UNSEEN, ang2pix

########################################################################################################################

def build_healpix_wcs(wcs: 'astropy.wcs.WCS') -> 'astropy.wcs.WCS':

    """
    Adjusts the provided World Coordinate System (WCS) object to perform proper HEALPix projections.

    Parameters
    ----------
    wcs : WCS
        The original WCS object.

    Returns
    -------
    WCS
        The modified WCS object.
    """

    ################################################################################################################

    result = wcs.copy()

    ################################################################################################################

    v = np.array([[
        wcs.wcs.crpix[0] - 0.5,
        wcs.wcs.crpix[1] - 0.5,
    ]])

    result.wcs.crval = wcs.all_pix2world(v, 0)[0]

    ################################################################################################################

    return result

########################################################################################################################

def image_to_healpix(wcs: 'astropy.wcs.WCS', nside: int, footprint: np.ndarray, rms_image: np.ndarray, bit_image: typing.Optional[np.ndarray] = None, rms_selection: float = 1.0e4, bit_selection: int = 0x00, show_progress_bar: bool = False) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Projects RMS (aka. noise) and bit (aka. data quality) image into HEALPix masks. **Nested ordering only.**

    Parameters
    ----------
    wcs : WCS
        The modified WCS object (see :func:`build_healpix_wcs`).
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the observed sky region.
    rms_image : np.ndarray
        2d image containing the RMS (aka. noise) information.
    bit_image : np.ndarray, default: **None**
        2d image containing the bit (aka. data quality) information.
    rms_selection : float, default: **1.0e4**
        Reject the pixel if RMS == 0 or RMS >= rms_selection.
    bit_selection : int, default: **0x00**
        Reject the pixel if (bit & bit_selection) != 0x00.
    show_progress_bar : bool, default = **False**
        Specifies whether to display a progress bar.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray, np.ndarray]
        First array contains the RMS (aka. noise) mask.
        Second array contains the bit (aka. data quality) mask.
        Third array contains the coverage (≡ fraction of observed sky) mask.
    """

    ####################################################################################################################
    # BUILD INDEX TABLE                                                                                                #
    ####################################################################################################################

    sorted_footprint_pixels = np.sort(footprint)

    sorted_footprint_indices = np.argsort(footprint)

    ####################################################################################################################
    # BUILD MASKS                                                                                                      #
    ####################################################################################################################

    if bit_image is None:
        bit_image_dtype = np.int32
    else:
        bit_image_dtype = bit_image.dtype

    ####################################################################################################################

    result_rms = np.zeros_like(footprint, dtype = rms_image.dtype)
    result_bit = np.zeros_like(footprint, dtype = bit_image_dtype)
    result_cov = np.zeros_like(footprint, dtype = rms_image.dtype)
    result_hit = np.zeros_like(footprint, dtype = rms_image.dtype)

    ####################################################################################################################

    x = np.arange(rms_image.shape[1], dtype = np.int64)

    y = np.empty(rms_image.shape[1], dtype = np.int64)

    ####################################################################################################################

    for j in tqdm.tqdm(range(rms_image.shape[0]), disable = not show_progress_bar):

        ################################################################################################################

        y.fill(j)

        ra, dec = wcs.all_pix2world(x, y, 0, ra_dec_order = True)

        pixels = ang2pix(nside, ra, dec, lonlat = True)

        ################################################################################################################

        if bit_image is None:
            _project1(result_rms, result_cov, result_hit, sorted_footprint_pixels, sorted_footprint_indices, pixels, rms_image[j], rms_selection)
        else:
            _project2(result_rms, result_bit, result_cov, result_hit, sorted_footprint_pixels, sorted_footprint_indices, pixels, rms_image[j], bit_image[j], rms_selection, bit_selection)

    ####################################################################################################################

    result_rms = np.where(result_hit != 0.0, result_rms / result_hit, UNSEEN)

    result_cov = np.where(result_hit != 0.0, result_cov / result_hit, 0.0000)

    ####################################################################################################################

    return np.sqrt(result_rms), result_bit, result_cov

########################################################################################################################

@nb.njit(fastmath = True)
def _project1(result_rms: np.ndarray, result_cov: np.ndarray, result_hit: np.ndarray, sorted_footprint_pixels: np.ndarray, sorted_footprint_indices: np.ndarray, pixels: np.ndarray, rms: np.ndarray, rms_selection: float) -> None:

    ####################################################################################################################

    sorted_indices = np.searchsorted(sorted_footprint_pixels, pixels)

    selected_idx_mask = sorted_footprint_pixels[sorted_indices] == pixels

    selected_idx = sorted_footprint_indices[sorted_indices[selected_idx_mask]]

    selected_rms = rms[selected_idx_mask]

    ####################################################################################################################

    for i in range(selected_idx.size):

        idx_i = selected_idx[i]
        rms_i = selected_rms[i]

        if 0.0 < rms_i < rms_selection:

            result_rms[idx_i] += rms_i ** 2

            result_cov[idx_i] += 1.0

        result_hit[idx_i] += 1.0

########################################################################################################################

@nb.njit(fastmath = True)
def _project2(result_rms: np.ndarray, result_bit: np.ndarray, result_cov: np.ndarray, result_hit: np.ndarray, sorted_footprint_pixels: np.ndarray, sorted_footprint_indices: np.ndarray, pixels: np.ndarray, rms: np.ndarray, bit: np.ndarray, rms_selection: float, bit_selection: int) -> None:

    ####################################################################################################################

    sorted_indices = np.searchsorted(sorted_footprint_pixels, pixels)

    selected_idx_mask = sorted_footprint_pixels[sorted_indices] == pixels

    selected_idx = sorted_footprint_indices[sorted_indices[selected_idx_mask]]

    selected_rms = rms[selected_idx_mask]
    selected_bit = bit[selected_idx_mask]

    ####################################################################################################################

    for i in range(selected_idx.size):

        idx_i = selected_idx[i]
        rms_i = selected_rms[i]
        bit_i = selected_bit[i]

        if 0.0 < rms_i < rms_selection:

            if (bit_i & bit_selection) == 0:

                result_rms[idx_i] += rms_i ** 2

                result_cov[idx_i] += 1.0

            result_bit[idx_i] |= bit_i

        result_hit[idx_i] += 1.0

########################################################################################################################
