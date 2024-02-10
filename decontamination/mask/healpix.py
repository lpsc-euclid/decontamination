# -*- coding: utf-8 -*-
########################################################################################################################

import tqdm
import typing

import numpy as np
import numba as nb

from ..hp import UNSEEN, ang2pix, nside2npix

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
    Projects RMS (aka. noise) and bit (aka. data quality) image into HEALPix masks.

    Parameters
    ----------
    wcs : WCS
        The modified WCS object (see :func:`build_healpix_wcs`).
    nside : int
        The HEALPix nside parameter. Must be less or equal to 16384.
    footprint : np.ndarray
        HEALPix indices of the observed region. **Nested ordering only.**
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

    if nside > 16384:

        raise ValueError('Nside must be <= 16384')

    ####################################################################################################################
    # BUILD INDEX TABLE                                                                                                #
    ####################################################################################################################

    npix = nside2npix(nside)

    index_table = np.full(npix, 0xFFFFFFFF, dtype = np.uint32)

    index_table[footprint] = np.arange(footprint.shape[0], dtype = np.uint32)

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
            _project1(result_rms, result_cov, result_hit, index_table, pixels, rms_image[j], rms_selection)
        else:
            _project2(result_rms, result_bit, result_cov, result_hit, index_table, pixels, rms_image[j], bit_image[j], rms_selection, bit_selection)

    ####################################################################################################################

    result_rms = np.where(result_hit != 0.0, result_rms / result_hit, UNSEEN)

    result_cov = np.where(result_hit != 0.0, result_cov / result_hit, 0.0000)

    ####################################################################################################################

    return np.sqrt(result_rms), result_bit, result_cov

########################################################################################################################

@nb.njit(fastmath = True)
def _project1(result_rms: np.ndarray, result_cov: np.ndarray, result_hit: np.ndarray, table: np.ndarray, pix: np.ndarray, rms: np.ndarray, rms_selection: float) -> None:

    ####################################################################################################################

    idx = table[pix]

    valid_idx_mask = idx != 0xFFFFFFFF

    valid_idx = idx[valid_idx_mask]
    valid_rms = rms[valid_idx_mask]

    ####################################################################################################################

    for i in range(valid_idx.size):

        idx_i = valid_idx[i]
        rms_i = valid_rms[i]

        if 0.0 < rms_i < rms_selection:

            result_rms[idx_i] += rms_i ** 2

            result_cov[idx_i] += 1.0

        result_hit[idx_i] += 1.0

########################################################################################################################

@nb.njit(fastmath = True)
def _project2(result_rms: np.ndarray, result_bit: np.ndarray, result_cov: np.ndarray, result_hit: np.ndarray, table: np.ndarray, pix: np.ndarray, rms: np.ndarray, bit: np.ndarray, rms_selection: float, bit_selection: int) -> None:

    ####################################################################################################################

    idx = table[pix]

    valid_idx_mask = idx != 0xFFFFFFFF

    valid_idx = idx[valid_idx_mask]
    valid_rms = rms[valid_idx_mask]
    valid_bit = bit[valid_idx_mask]

    ####################################################################################################################

    for i in range(valid_idx.size):

        idx_i = valid_idx[i]
        rms_i = valid_rms[i]
        bit_i = valid_bit[i]

        if 0.0 < rms_i < rms_selection:

            if (bit_i & bit_selection) == 0:

                result_rms[idx_i] += rms_i ** 2

                result_cov[idx_i] += 1.0

            result_bit[idx_i] |= bit_i

        result_hit[idx_i] += 1.0

########################################################################################################################
