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

import concurrent.futures

from .wcs import WCS
from ..hp import UNSEEN, ang2pix

########################################################################################################################

def rms_bit_to_healpix(wcs: WCS, nside: int, footprint: np.ndarray, rms_image: np.ndarray, bit_image: typing.Optional[np.ndarray] = None, rms_cutoff: float = 1.0e4, bit_selection: int = 0, n_threads: int = 1, show_progress_bar: bool = False) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Projects RMS (aka. noise) and bit (aka. data quality) images into a HEALPix footprint.

    .. warning::
        Nested ordering only.

    Parameters
    ----------
    wcs : WCS
        The WCS in HEALPix mode.
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the observed sky region.
    rms_image : np.ndarray
        2d image containing the RMS (aka. noise) information.
    bit_image : np.ndarray, default: **None**
        2d image containing the bit (aka. data quality) information.
    rms_cutoff : float, default: **1.0e4**
        Reject the pixel if RMS <= 0 or RMS >= rms_cutoff.
    bit_selection : int, default: **0x00**
        Reject the pixel if (bit & bit_selection) != 0x00.
    n_threads : int, default: 1
        Number of threads.
    show_progress_bar : bool, default = **False**
        Specifies whether to display a progress bar.

    Returns
    -------
    np.ndarray
        First array contains the RMS (aka. noise) mask.
    np.ndarray
        Second array contains the bit (aka. data quality) mask.
    np.ndarray
        Third array contains the coverage (≡ fraction of observed sky) mask.
    """

    ####################################################################################################################
    # BUILD INDEX TABLE                                                                                                #
    ####################################################################################################################

    sorted_footprint_pixels = np.sort(footprint)

    sorted_footprint_indices = np.argsort(footprint)

    ####################################################################################################################
    # BUILD RMS, BIT AND COVERAGE MASKS                                                                                #
    ####################################################################################################################

    if bit_image is None:

        bit_image = np.zeros_like(rms_image, dtype = np.uint32)

    ################################################################################################################

    result_rms = np.zeros_like(footprint, dtype = rms_image.dtype)
    result_cov = np.zeros_like(footprint, dtype = rms_image.dtype)
    result_hit = np.zeros_like(footprint, dtype = rms_image.dtype)
    result_bit = np.zeros_like(footprint, dtype = bit_image.dtype)

    ####################################################################################################################

    rows_per_thread = rms_image.shape[0] // n_threads

    with concurrent.futures.ThreadPoolExecutor(max_workers = n_threads) as executor:

        futures = []

        for i in range(n_threads):

            j2 = (i + 1) * rows_per_thread \
                if i < n_threads - 1 else rms_image.shape[0]
            j1 = (i + 0) * rows_per_thread

            futures.append(executor.submit(
                _worker1,
                wcs,
                nside,
                footprint,
                sorted_footprint_pixels,
                sorted_footprint_indices,
                j1, j2,
                rms_image, bit_image,
                rms_cutoff, bit_selection,
                show_progress_bar
            ))

        for future in concurrent.futures.as_completed(futures):

            tmp_rms, tmp_cov, tmp_hit, tmp_bit = future.result()

            result_rms += tmp_rms
            result_cov += tmp_cov
            result_bit |= tmp_bit
            result_hit += tmp_hit

    ####################################################################################################################
    # NORMALIZE RMS, BIT AND COVERAGE MASKS                                                                            #
    ####################################################################################################################

    has_cov = result_cov > 0.0

    ####################################################################################################################

    old_settings = np.seterr(divide = 'ignore', invalid = 'ignore')

    result_rms = np.where(has_cov, result_rms / result_cov, UNSEEN)
    result_bit = np.where(has_cov, result_bit, 0xFFFFFFFF)
    result_cov = np.where(has_cov, result_cov / result_hit, 0.0000)

    np.seterr(**old_settings)

    ####################################################################################################################

    result_rms[has_cov] = np.sqrt(result_rms[has_cov])

    ####################################################################################################################

    return result_rms, result_bit, result_cov

########################################################################################################################

def image_to_healpix(wcs: WCS, nside: int, footprint: np.ndarray, xxx_image: np.ndarray, xxx_image_scale: float = 1.0, quadratic: bool = False, n_threads: int = 1, show_progress_bar: bool = False) -> np.ndarray:

    """
    Projects the given image into a HEALPix footprint.

    .. warning::
        Nested ordering only.

    Parameters
    ----------
    wcs : WCS
        The WCS in HEALPix mode.
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the observed sky region.
    xxx_image : np.ndarray
        2d image to be projected into the footprint.
    xxx_image_scale : int, default: **1.0**
        Scale so that the image size coincides with the WCS (>= 1.0).
    quadratic : bool, default: False
        ???.
    n_threads : int, default: 1
        Number of threads.
    show_progress_bar : bool, default = **False**
        Specifies whether to display a progress bar.

    Returns
    -------
    np.ndarray
        The resulting HEALPix mask.
    """

    if xxx_image_scale < 1.0:

        raise ValueError('The image scale must be greater than or equal to 1.0')

    ####################################################################################################################
    # BUILD INDEX TABLE                                                                                                #
    ####################################################################################################################

    sorted_footprint_pixels = np.sort(footprint)

    sorted_footprint_indices = np.argsort(footprint)

    ####################################################################################################################
    # BUILD MASKS                                                                                                      #
    ####################################################################################################################

    if quadratic:

        xxx_image = np.square(xxx_image)

    ####################################################################################################################

    result_xxx = np.zeros_like(footprint, dtype = xxx_image.dtype)
    result_hit = np.zeros_like(footprint, dtype = xxx_image.dtype)

    ####################################################################################################################

    rows_per_thread = xxx_image.shape[0] // n_threads

    with concurrent.futures.ThreadPoolExecutor(max_workers = n_threads) as executor:

        futures = []

        for i in range(n_threads):

            j2 = (i + 1) * rows_per_thread \
                if i < n_threads - 1 else xxx_image.shape[0]
            j1 = (i + 0) * rows_per_thread

            futures.append(executor.submit(
                _worker2,
                wcs,
                nside,
                footprint,
                sorted_footprint_pixels,
                sorted_footprint_indices,
                j1, j2,
                xxx_image,
                xxx_image_scale,
                show_progress_bar
            ))

        for future in concurrent.futures.as_completed(futures):

            tmp_xxx, tmp_hit = future.result()

            result_xxx += tmp_xxx
            result_hit += tmp_hit

    ####################################################################################################################
    # NORMALIZE MASKS                                                                                                  #
    ####################################################################################################################

    has_hit = result_hit > 0.0

    ####################################################################################################################

    old_settings = np.seterr(divide = 'ignore', invalid = 'ignore')

    result_xxx = np.where(has_hit, result_xxx / result_hit, UNSEEN)

    np.seterr(**old_settings)

    ####################################################################################################################

    if quadratic:

        result_xxx[has_hit] = np.sqrt(result_xxx[has_hit])

    ####################################################################################################################

    return result_xxx

########################################################################################################################
# WORKERS                                                                                                              #
########################################################################################################################

# noinspection DuplicatedCode
def _worker1(wcs: WCS, nside: int, footprint, sorted_footprint_pixels, sorted_footprint_indices, j1, j2, rms_image, bit_image, rms_cutoff, bit_selection, show_progress_bar):

    ####################################################################################################################

    result_rms = np.zeros_like(footprint, dtype = rms_image.dtype)
    result_cov = np.zeros_like(footprint, dtype = rms_image.dtype)
    result_bit = np.zeros_like(footprint, dtype = bit_image.dtype)
    result_hit = np.zeros_like(footprint, dtype = rms_image.dtype)

    ####################################################################################################################

    x = np.arange(rms_image.shape[1], dtype = np.int64)

    y = np.empty(rms_image.shape[1], dtype = np.int64)

    # = 1.0000000000000 * x

    for j in tqdm.tqdm(range(j1, j2), disable = not show_progress_bar):

        ################################################################################################################

        y.fill(j)

        ra, dec = wcs.all_pix2world(x, y, 0, ra_dec_order = True)

        pixels = ang2pix(nside, ra, dec, lonlat = True)

        ################################################################################################################

        _project1(
            result_rms,
            result_cov,
            result_bit,
            result_hit,
            sorted_footprint_pixels,
            sorted_footprint_indices,
            pixels,
            rms_image[j],
            bit_image[j],
            rms_cutoff,
            bit_selection
        )

    ####################################################################################################################

    return result_rms, result_cov, result_bit, result_hit

########################################################################################################################

# noinspection DuplicatedCode
def _worker2(wcs: WCS, nside: int, footprint, sorted_footprint_pixels, sorted_footprint_indices, j1, j2, xxx_image, xxx_image_scale, show_progress_bar):

    ####################################################################################################################

    result_xxx = np.zeros_like(footprint, dtype = xxx_image.dtype)
    result_hit = np.zeros_like(footprint, dtype = xxx_image.dtype)

    ####################################################################################################################

    x = np.arange(xxx_image.shape[1], dtype = np.int64)

    y = np.empty(xxx_image.shape[1], dtype = np.int64)

    x = xxx_image_scale * x

    for j in tqdm.tqdm(range(j1, j2), disable = not show_progress_bar):

        ################################################################################################################

        y.fill(xxx_image_scale * j)

        ra, dec = wcs.all_pix2world(x, y, 0, ra_dec_order = True)

        pixels = ang2pix(nside, ra, dec, lonlat = True)

        ################################################################################################################

        _project2(
            result_xxx,
            result_hit,
            sorted_footprint_pixels,
            sorted_footprint_indices,
            pixels,
            xxx_image[j]
        )

    ####################################################################################################################

    return result_xxx, result_hit

########################################################################################################################
# PROJECTORS                                                                                                           #
########################################################################################################################

@nb.njit(fastmath = True)
def _project1(result_rms: np.ndarray, result_cov: np.ndarray, result_bit: np.ndarray, result_hit: np.ndarray, sorted_footprint_pixels: np.ndarray, sorted_footprint_indices: np.ndarray, pixels: np.ndarray, rms: np.ndarray, bit: np.ndarray, rms_cutoff: float, bit_selection: int) -> None:

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

        if 0.0 < rms_i < rms_cutoff:

            if (bit_i & bit_selection) == 0:

                result_rms[idx_i] += rms_i ** 2

                result_cov[idx_i] += 1.000

            result_bit[idx_i] |= bit_i

        result_hit[idx_i] += 1.000

########################################################################################################################

@nb.njit(fastmath = True)
def _project2(result_xxx: np.ndarray, result_cov: np.ndarray, sorted_footprint_pixels: np.ndarray, sorted_footprint_indices: np.ndarray, pixels: np.ndarray, xxx: np.ndarray) -> None:

    ####################################################################################################################

    sorted_indices = np.searchsorted(sorted_footprint_pixels, pixels)

    selected_idx_mask = sorted_footprint_pixels[sorted_indices] == pixels

    selected_idx = sorted_footprint_indices[sorted_indices[selected_idx_mask]]

    selected_xxx = xxx[selected_idx_mask]

    ####################################################################################################################

    for i in range(selected_idx.size):

        idx_i = selected_idx[i]
        xxx_i = selected_xxx[i]

        result_xxx[idx_i] += xxx_i

        result_cov[idx_i] += 1.000

########################################################################################################################
