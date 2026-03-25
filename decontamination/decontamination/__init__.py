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

from ..logging import logger
from ..algo import dataset_to_generator_builder

########################################################################################################################

@nb.njit()
def estimate_n_temp_bins(n_bins: int, n_vectors: int) -> int:

    ####################################################################################################################

    min_bins = 32 * n_bins
    max_bins = 256 * n_bins

    ####################################################################################################################

    if n_vectors <= 0:

        return max(min_bins, 1)

    ####################################################################################################################

    return int(max(min_bins, min(max_bins, 2.0 * np.sqrt(float(n_vectors)))))

########################################################################################################################

@nb.njit()
def _build_equal_area_edges(result_edges: np.ndarray, hits: np.ndarray, minimum: float, maximum: float, n_bins: int) -> None:

    ####################################################################################################################

    result_edges[0x0000] = minimum
    result_edges[n_bins] = maximum

    if not (maximum > minimum):

        result_edges[1: n_bins] = minimum

        return

    ####################################################################################################################

    total_hits = np.sum(hits)

    if not (total_hits > 0.0):

        result_edges[1: n_bins] = minimum

        return

    ####################################################################################################################

    idx = 0

    acc_hits = 0.0

    bin_hits = total_hits / n_bins

    ####################################################################################################################

    for j in range(hits.shape[0]):

        ################################################################################################################

        tmp_hits = hits[j]

        if tmp_hits <= 0.0:

            continue

        ################################################################################################################

        take_hits = 0.0

        while take_hits < tmp_hits and idx < n_bins - 1:

            need_hits = bin_hits - acc_hits
            left_hits = tmp_hits - take_hits

            if left_hits < need_hits:

                acc_hits += left_hits
                take_hits = tmp_hits

            else:

                take_hits += need_hits

                result_edges[idx + 1] = ((j + take_hits / tmp_hits) / hits.shape[0]) * (maximum - minimum) + minimum

                acc_hits = 0.0

                idx += 1

    ####################################################################################################################

    result_edges[idx + 1: n_bins] = maximum

########################################################################################################################

def _accumulate_global_stats(systematics: np.ndarray, dim: typing.Optional[int], n_vectors: int, sum1: typing.Optional[np.ndarray], sum2: typing.Optional[np.ndarray], minima: typing.Optional[np.ndarray], maxima: typing.Optional[np.ndarray]) -> typing.Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    ####################################################################################################################

    if systematics.ndim != 2:

        raise ValueError('Chunks must have shape (dim, n_vectors)')

    ####################################################################################################################

    if dim is None:

        dim = systematics.shape[0]

        n_vectors = 0

        sum1 = np.zeros(dim, dtype = np.float64)
        sum2 = np.zeros(dim, dtype = np.float64)

        minima = np.full(dim, +np.inf, dtype = np.float64)
        maxima = np.full(dim, -np.inf, dtype = np.float64)

    elif systematics.shape[0] != dim:

        raise ValueError('Inconsistent number of systematics across chunks')

    ####################################################################################################################

    valid_mask = np.all(np.isfinite(systematics), axis = 0)

    ####################################################################################################################

    if np.any(valid_mask):

        valid_systematics = systematics[:, valid_mask].astype(np.float64, copy = False)

        n_vectors += valid_systematics.shape[1]

        for i in range(dim):

            systematic = valid_systematics[i]

            sum1[i] += np.sum(systematic ** 1)
            sum2[i] += np.sum(systematic ** 2)

            minimum = np.min(systematic)
            maximum = np.max(systematic)

            if minima[i] > minimum:
                minima[i] = minimum

            if maxima[i] < maximum:
                maxima[i] = maximum

    ####################################################################################################################

    return dim, n_vectors, sum1, sum2, minima, maxima, valid_mask

########################################################################################################################

def _accumulate_temporary_histograms(systematics: np.ndarray, dim: int, tmp_n_bins: np.ndarray, minima: np.ndarray, maxima: np.ndarray, hits: typing.List[np.ndarray]) -> None:

    ####################################################################################################################

    valid_mask = np.all(np.isfinite(systematics), axis = 0)

    ####################################################################################################################

    if np.any(valid_mask):

        valid_systematics = systematics[:, valid_mask]

        for i in range(dim):

            systematic = valid_systematics[i]

            if maxima[i] > minima[i]:

                hits[i] += np.histogram(systematic.astype(np.float64, copy = False), bins = tmp_n_bins[i], range = (minima[i], maxima[i]))[0]

########################################################################################################################

# noinspection DuplicatedCode
def _accumulate_bin_centers(systematics: np.ndarray, dim: int, n_bins: int, minima: np.ndarray, maxima: np.ndarray, result_edges: np.ndarray, result_sum: np.ndarray, result_count: np.ndarray) -> None:

    ####################################################################################################################

    valid_mask = np.all(np.isfinite(systematics), axis = 0)

    ####################################################################################################################

    if np.any(valid_mask):

        valid_systematics = systematics[:, valid_mask].astype(np.float64, copy = False)

        for i in range(dim):

            systematic = valid_systematics[i]

            if maxima[i] > minima[i]:

                indices = np.clip(np.searchsorted(result_edges[i], systematic, side = 'right') - 1, 0, n_bins - 1)

                result_sum[i] += np.bincount(indices, weights = systematic, minlength = n_bins)
                result_count[i] += np.bincount(indices, weights = None, minlength = n_bins)

            else:

                result_sum[i] += np.sum(systematic)
                result_count[i] += systematic.size

########################################################################################################################

def _compute_exact_equal_area_edges_and_centers(systematics: np.ndarray, n_bins: int) -> typing.Tuple[np.ndarray, np.ndarray]:

    ####################################################################################################################

    dim = systematics.shape[0]

    ####################################################################################################################

    result_edges = np.empty((dim, n_bins + 1), dtype = np.float64)
    result_centers = np.empty((dim, n_bins + 0), dtype = np.float64)

    ####################################################################################################################

    for i in range(dim):

        ################################################################################################################

        systematic = np.sort(systematics[i].astype(np.float64, copy = False))

        chunks = np.array_split(systematic, n_bins)

        ################################################################################################################

        result_edges[i, 0x0000] = systematic[0]
        result_edges[i, n_bins] = systematic[-1]

        ################################################################################################################

        for j in range(n_bins):

            chunk = chunks[j]

            if chunk.size > 0:

                result_centers[i, j] = np.mean(chunk)

            else:

                result_centers[i, j] = np.nan

            if j + 1 < n_bins:

                next_chunk = chunks[j + 1]

                if next_chunk.size > 0:

                    result_edges[i, j + 1] = next_chunk[0]

                else:

                    result_edges[i, j + 1] = systematic[-1]

    ####################################################################################################################

    return result_edges, result_centers

########################################################################################################################

def compute_equal_area_binning_and_statistics(systematics: typing.Union[np.ndarray, typing.Callable], n_bins: int, n_temp_bins: typing.Optional[int] = None, exact: bool = False, show_progress_bar: bool = False) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:

    """
    Computes equal-area binning and global statistics for a set of systematics.

    The input data can be provided either as a full array or as a generator builder yielding chunks. Two modes are available:

    - **exact** = **True**: exact equal-area binning, requiring all valid vectors to be kept in memory.
    - **exact** = **False**: chunk-based approximation of the bin edges using temporary histograms.

    .. note::
        A vector of systematics is discarded if at least one of its components is not finite.

    Parameters
    ----------
    systematics : typing.Union[np.ndarray, typing.Callable]
        Input array of shape :math:`(\\mathrm{dim},N_\\mathrm{vectors})` or generator builder.
    n_bins : int
        Number of bins to build for each systematic.
    n_temp_bins : typing.Optional[int], default: **None** ≡ :math:`\\max\\left(32\\,n_\\mathrm{bins},\\min\\left(256\\,n_\\mathrm{bins},\\,2\\sqrt{n_\\mathrm{vectors}}\\right)\\right)`
        Number of temporary histogram bins used when **exact** = **False**.
    exact : bool, default: **False**
        If **True**, computes exact equal-area binning. If **False**, computes chunk-based approximate edges.
    show_progress_bar : bool, default: **False**
        Specifies whether to display a progress bar.

    Returns
    -------
    result_edges : np.ndarray
        Array of shape :math:`(\\mathrm{dim},n_\\mathrm{bins}+1)` containing the bin edges for each systematic.
    result_centers : np.ndarray
        Array of shape :math:`(\\mathrm{dim},n_\\mathrm{bins})` containing the mean value in each bin.
    result_minima : np.ndarray
        Array of shape :math:`(\\mathrm{dim},)` containing the minimum value of each systematic.
    result_maxima : np.ndarray
        Array of shape :math:`(\\mathrm{dim},)` containing the maximum value of each systematic.
    result_means : np.ndarray
        Array of shape :math:`(\\mathrm{dim},)` containing the mean value of each systematic.
    result_rmss : np.ndarray
        Array of shape :math:`(\\mathrm{dim},)` containing the root-mean-square of each systematic.
    result_stds : np.ndarray
        Array of shape :math:`(\\mathrm{dim},)` containing the standard deviation of each systematic.
    result_n_vectors : int
        Number of valid vectors used in the computation.
    """

    ####################################################################################################################

    if n_bins <= 0:

        raise ValueError('`n_bins` must be > 0')

    ####################################################################################################################

    generator_builder = dataset_to_generator_builder(systematics)

    if generator_builder is None:

        raise ValueError('`systematics` must not be None')

    ####################################################################################################################
    # PASS 1: COMPUTE GLOBAL STATISTICS                                                                                #
    ####################################################################################################################

    dim = None
    result_n_vectors = 0x00

    sum1 = None
    sum2 = None

    result_minima = None
    result_maxima = None

    ####################################################################################################################

    n_iters = 0

    generator = generator_builder()

    exact_chunks = [] if exact else None

    for systematics_chunk in tqdm.tqdm(generator(), total = None, disable = not show_progress_bar):

        dim, result_n_vectors, sum1, sum2, result_minima, result_maxima, valid_mask = _accumulate_global_stats(
            systematics_chunk,
            dim,
            result_n_vectors,
            sum1,
            sum2,
            result_minima,
            result_maxima
        )

        ################################################################################################################

        if exact and np.any(valid_mask):

            exact_chunks.append(systematics_chunk[:, valid_mask].astype(np.float64, copy = False))

        ################################################################################################################

        n_iters += 1

    ####################################################################################################################

    if dim is None:

        raise ValueError('Empty dataset')

    if result_n_vectors <= 0:

        raise ValueError('No fully finite vector found')

    ####################################################################################################################

    result_means = np.divide(
        sum1,
        result_n_vectors,
        out = np.full(dim, np.nan, dtype = np.float64),
        where = result_n_vectors > 0
    )

    result_rmss = np.sqrt(np.divide(
        sum2,
        result_n_vectors,
        out = np.full(dim, np.nan, dtype = np.float64),
        where = result_n_vectors > 0
    ))

    variances = np.divide(
        sum2 - (sum1 * sum1) / result_n_vectors,
        result_n_vectors - 1,
        out = np.full(dim, np.nan, dtype = np.float64),
        where = result_n_vectors > 1
    )

    ####################################################################################################################

    result_stds = np.sqrt(np.maximum(0.0, variances))

    ####################################################################################################################
    # EXACT MODE                                                                                                       #
    ####################################################################################################################

    if exact:

        ################################################################################################################

        if n_bins > result_n_vectors:

            raise ValueError('`n_bins` must be <= number of valid vectors when `exact` is True')

        if len(exact_chunks) == 0:

            raise ValueError('No valid vectors available for exact mode')

        ################################################################################################################

        result_edges, result_centers = _compute_exact_equal_area_edges_and_centers(
            np.concatenate(exact_chunks, axis = 1),
            n_bins
        )

        return (
            result_edges,
            result_centers,
            result_minima,
            result_maxima,
            result_means,
            result_rmss,
            result_stds,
            result_n_vectors
        )

    ####################################################################################################################
    # PASS 2: BUILD TEMPORARY HISTOGRAMS                                                                               #
    ####################################################################################################################

    if n_temp_bins is None:

        tmp_n_bins = np.full(dim, estimate_n_temp_bins(n_bins, int(result_n_vectors)), dtype = np.int64)

    elif n_temp_bins >= n_bins:

        tmp_n_bins = np.full(dim, n_temp_bins, dtype = np.int64)

    else:

        raise ValueError('`n_temp_bins` must be >= `n_bins`')

    ####################################################################################################################

    logger.info(f'Using {tmp_n_bins[0]} temporary bins for equal-area approximation')

    ####################################################################################################################

    hits = [np.zeros(tmp_n_bins[i], dtype = np.float64) for i in range(dim)]

    ####################################################################################################################

    generator = generator_builder()

    for systematics_chunk in tqdm.tqdm(generator(), total = n_iters, disable = not show_progress_bar):

        _accumulate_temporary_histograms(
            systematics_chunk,
            dim,
            tmp_n_bins,
            result_minima,
            result_maxima,
            hits
        )

    ####################################################################################################################
    # BUILD FINAL EDGES                                                                                                #
    ####################################################################################################################

    result_edges = np.empty((dim, n_bins + 1), dtype = np.float64)

    ####################################################################################################################

    for i in range(dim):

        _build_equal_area_edges(
            result_edges[i],
            hits[i],
            float(result_minima[i]),
            float(result_maxima[i]),
            n_bins
        )

    ####################################################################################################################
    # PASS 3: COMPUTE EXACT BIN CENTERS                                                                                #
    ####################################################################################################################

    tmp_sum = np.zeros((dim, n_bins), dtype = np.float64)
    tmp_count = np.zeros((dim, n_bins), dtype = np.int64)

    ####################################################################################################################

    generator = generator_builder()

    for systematics_chunk in tqdm.tqdm(generator(), total = n_iters, disable = not show_progress_bar):

        _accumulate_bin_centers(
            systematics_chunk,
            dim,
            n_bins,
            result_minima,
            result_maxima,
            result_edges,
            tmp_sum,
            tmp_count
        )

    ####################################################################################################################

    result_centers = np.divide(
        tmp_sum,
        tmp_count,
        out = np.full((dim, n_bins), np.nan, dtype = np.float64), where = tmp_count > 0
    )

    ####################################################################################################################

    return (
        result_edges,
        result_centers,
        result_minima,
        result_maxima,
        result_means,
        result_rmss,
        result_stds,
        result_n_vectors
    )

########################################################################################################################

# noinspection DuplicatedCode
def _accumulate_equal_area_correlation_chunk(systematics: np.ndarray, galaxy_number_density: np.ndarray, dim: typing.Optional[int], n_bins: int, edges: np.ndarray, result_sum_density: typing.Optional[np.ndarray], result_num_pixels: typing.Optional[np.ndarray], result_total_density: typing.Optional[np.ndarray], result_total_pixels: typing.Optional[np.ndarray]) -> typing.Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    ####################################################################################################################

    if systematics.ndim != 2:

        raise ValueError('`systematics` chunks must have shape (dim, n_vectors)')

    ####################################################################################################################

    if galaxy_number_density.ndim != 1:

        raise ValueError('`galaxy_number_density` chunks must have shape (n_vectors,)')

    ####################################################################################################################

    if systematics.shape[1] != galaxy_number_density.shape[0]:

        raise ValueError('`systematics` and `galaxy_number_density` chunks must be aligned')

    ####################################################################################################################

    if dim is None:

        dim = systematics.shape[0]

        result_sum_density = np.zeros((dim, n_bins), dtype = np.float64)
        result_num_pixels = np.zeros((dim, n_bins), dtype = np.int64)

        result_total_density = np.zeros(dim, dtype = np.float64)
        result_total_pixels = np.zeros(dim, dtype = np.int64)

    elif systematics.shape[0] != dim:

        raise ValueError('Inconsistent number of systematics across chunks')

    ####################################################################################################################

    density = galaxy_number_density.astype(np.float64, copy = False)

    ####################################################################################################################

    for i in range(dim):

        systematic = systematics[i].astype(np.float64, copy = False)

        valid_mask = np.isfinite(systematic) & np.isfinite(density)

        if np.any(valid_mask):

            ############################################################################################################

            valid_systematic = systematic[valid_mask]
            valid_density = density[valid_mask]

            ############################################################################################################

            indices = np.clip(np.searchsorted(edges[i], valid_systematic, side = 'right') - 1, 0, n_bins - 1)

            ############################################################################################################

            result_sum_density[i] += np.bincount(indices, weights = valid_density, minlength = n_bins)
            result_num_pixels[i] += np.bincount(indices, weights = None, minlength = n_bins)

            ############################################################################################################

            result_total_density[i] += np.sum(valid_density)
            result_total_pixels[i] += valid_density.size

    ####################################################################################################################

    return dim, result_sum_density, result_num_pixels, result_total_density, result_total_pixels

########################################################################################################################

def compute_equal_area_correlation(systematics: typing.Union[np.ndarray, typing.Callable], galaxy_number_density: typing.Union[np.ndarray, typing.Callable], edges: np.ndarray) -> np.ndarray:

    """
    Computes the equal-area correlation between systematics and a galaxy number density.

    For each systematic and each bin, the returned value is:

    .. math::
        C_i=\\frac{\\langle n_\\mathrm{gal}\\rangle_{\\mathrm{bin},i}}{\\langle n_\\mathrm{gal}\\rangle}

    where :math:`\\langle n_\\mathrm{gal} \\rangle_{\\mathrm{bin},i}` is the mean galaxy number density in bin
    :math:`i`, and :math:`\\langle n_\\mathrm{gal} \\rangle` is the global mean galaxy number density
    computed on the valid footprint for the considered systematic.

    .. note::
        A pair `(systematics, galaxy_number_density)` is discarded if at least one of the two values is not finite.

    Parameters
    ----------
    systematics : typing.Union[np.ndarray, typing.Callable]
        Systematics of shape :math:`(\\mathrm{dim},N_\\mathrm{vectors})` or generator builder.
    galaxy_number_density : typing.Union[np.ndarray, typing.Callable]
        Galaxy number density of shape :math:`(N_\\mathrm{vectors},)` or generator builder.
    edges : np.ndarray
        Array of shape :math:`(\\mathrm{dim},n_\\mathrm{bins}+1)` containing the bin edges for each systematic.

    Returns
    -------
    np.ndarray
        Array of shape :math:`(\\mathrm{dim},n_\\mathrm{bins})` containing the correlation values.
    """

    ####################################################################################################################

    systematics_generator_builder = dataset_to_generator_builder(systematics)
    galaxy_number_density_generator_builder = dataset_to_generator_builder(galaxy_number_density)

    ####################################################################################################################

    if systematics_generator_builder is None:

        raise ValueError('`systematics` must not be None')

    ####################################################################################################################

    if galaxy_number_density_generator_builder is None:

        raise ValueError('`galaxy_number_density` must not be None')

    ####################################################################################################################

    n_bins = edges.shape[1] - 1

    if edges.ndim != 2 or n_bins <= 0:

        raise ValueError('`edges` must have shape (dim, n_bins + 1) with n_bins > 0')

    ####################################################################################################################

    dim = None

    sum_density = None
    num_pixels = None

    total_density = None
    total_pixels = None

    ####################################################################################################################

    systematics_generator = systematics_generator_builder()
    galaxy_number_density_generator = galaxy_number_density_generator_builder()

    ####################################################################################################################

    for systematics_chunk, galaxy_number_density_chunk in zip(systematics_generator(), galaxy_number_density_generator()):

        dim, sum_density, num_pixels, total_density, total_pixels = _accumulate_equal_area_correlation_chunk(
            systematics_chunk,
            galaxy_number_density_chunk,
            dim,
            n_bins,
            edges,
            sum_density,
            num_pixels,
            total_density,
            total_pixels
        )

    ####################################################################################################################

    if dim is None:

        raise ValueError('Empty dataset')

    ####################################################################################################################

    if edges.shape[0] != dim:

        raise ValueError('`edges` first dimension must match the number of systematics')

    ####################################################################################################################

    global_mean_density = np.divide(
        total_density,
        total_pixels,
        out = np.full(dim, np.nan, dtype = np.float64), where = total_pixels > 0
    )

    mean_density_per_bin = np.divide(
        sum_density,
        num_pixels,
        out = np.full((dim, n_bins), np.nan, dtype = np.float64), where = num_pixels > 0
    )

    ####################################################################################################################

    return np.divide(
        mean_density_per_bin,
        global_mean_density[:, np.newaxis],
        out = np.full((dim, n_bins), np.nan, dtype = np.float64),
        where = np.isfinite(global_mean_density[:, np.newaxis]) & (global_mean_density[:, np.newaxis] != 0.0)
    )

########################################################################################################################
