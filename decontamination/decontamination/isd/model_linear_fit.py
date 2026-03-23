# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-Perez <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import tqdm
import typing

import numpy as np
import numba as nb

from ..algo import dataset_to_generator_builder
from ..generator import generator_number_density

########################################################################################################################

# noinspection PyPep8Naming
class Decontamination_Abstract(object):

    """
    Systematics decontamination (abstract class).
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, coverage: np.ndarray, footprint_systematics: typing.Union[np.ndarray, typing.Callable], galaxy_number_density: typing.Union[np.ndarray, typing.Callable]):

        ################################################################################################################

        self._nside = nside
        self._footprint = footprint
        self._coverage = coverage
        self._footprint_systematics = footprint_systematics
        self._galaxy_number_density = galaxy_number_density

        ################################################################################################################

        self._corrected_galaxy_number_density = galaxy_number_density / coverage

    ####################################################################################################################

    @property
    def nside(self) -> int:

        """Nside."""

        return self._nside

    ####################################################################################################################

    @property
    def footprint(self) -> int:

        """Footprint."""

        return self._footprint

    ####################################################################################################################

    @property
    def coverage(self) -> int:

        """Coverage."""

        return self._coverage

    ####################################################################################################################

    @property
    def footprint_systematics(self) -> int:

        """Footprint systematics."""

        return self._footprint_systematics

    ####################################################################################################################

    @property
    def galaxy_number_density(self) -> int:

        """Galaxy number density."""

        return self._galaxy_number_density

    ####################################################################################################################

    @property
    def corrected_galaxy_number_density(self) -> int:

        """Coverage-corrected galaxy number density."""

        return self._corrected_galaxy_number_density

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _compute_default_temp_n_bins(n_bins: int, count: int) -> int:

        ################################################################################################################

        min_bins = 32 * n_bins
        max_bins = 256 * n_bins

        ################################################################################################################

        if count <= 0:

            return max(min_bins, 1)

        ################################################################################################################

        return int(max(min_bins, min(max_bins, 2.0 * np.sqrt(float(count)))))

    ####################################################################################################################

    @staticmethod
    def _accumulate_global_stats(vectors: np.ndarray, dim: typing.Optional[int], counts: typing.Optional[np.ndarray], sum1: typing.Optional[np.ndarray], sum2: typing.Optional[np.ndarray], minima: typing.Optional[np.ndarray], maxima: typing.Optional[np.ndarray]) -> typing.Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        if vectors.ndim != 2:

            raise ValueError('Chunks must have shape (dim, n_vectors)')

        ################################################################################################################

        if dim is None:

            dim = vectors.shape[0]

            counts = np.zeros(dim, dtype = np.int64)

            sum1 = np.zeros(dim, dtype = np.float64)
            sum2 = np.zeros(dim, dtype = np.float64)

            minima = np.full(dim, +np.inf, dtype = np.float64)
            maxima = np.full(dim, -np.inf, dtype = np.float64)

        elif vectors.shape[0] != dim:

            raise ValueError('Inconsistent number of systematics across chunks')

        ################################################################################################################

        for i in range(dim):

            systematic = vectors[i]

            valid = systematic[np.isfinite(systematic)]

            if valid.size > 0:

                valid = valid.astype(np.float64, copy = False)

                counts[i] += valid.size

                sum1[i] += np.sum(    valid    )
                sum2[i] += np.sum(valid * valid)

                minimum = np.min(valid)
                maximum = np.max(valid)

                if minima[i] > minimum:
                    minima[i] = minimum

                if maxima[i] < maximum:
                    maxima[i] = maximum

        ################################################################################################################

        return dim, counts, sum1, sum2, minima, maxima

    ####################################################################################################################

    @staticmethod
    def _accumulate_temporary_histograms(vectors: np.ndarray, dim: int, tmp_n_bins: np.ndarray, minima: np.ndarray, maxima: np.ndarray, hits: typing.List[np.ndarray]) -> None:

        ################################################################################################################

        for i in range(dim):

            systematic = vectors[i]

            valid = systematic[np.isfinite(systematic)]

            if valid.size > 0 and (maxima[i] > minima[i]):

                hits[i] += np.histogram(valid.astype(np.float64, copy = False), bins = tmp_n_bins[i], range = (minima[i], maxima[i]))[0]

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _build_equal_sky_area_edges(result_edges: np.ndarray, hits: np.ndarray, minimum: float, maximum: float, n_bins: int) -> None:

        ################################################################################################################

        result_edges[0x0000] = minimum
        result_edges[n_bins] = maximum

        if not (maximum > minimum):

            result_edges[1: n_bins] = minimum

            return

        ################################################################################################################

        total_hits = np.sum(hits)

        if not (total_hits > 0.0):

            result_edges[1: n_bins] = minimum

            return

        ################################################################################################################

        idx = 0

        acc_hits = 0.0

        bin_hits = total_hits / n_bins

        ################################################################################################################

        for j in range(hits.shape[0]):

            ############################################################################################################

            tmp_hits = hits[j]

            if tmp_hits > 0.0:

                ########################################################################################################

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

        ################################################################################################################

        result_edges[idx + 1: n_bins] = maximum

    ####################################################################################################################

    @staticmethod
    def _accumulate_bin_centers(vectors: np.ndarray, dim: int, n_bins: int, minima: np.ndarray, maxima: np.ndarray, result_edges: np.ndarray, result_sum: np.ndarray, result_count: np.ndarray) -> None:

        ################################################################################################################

        for i in range(dim):

            systematic = vectors[i]

            valid = systematic[np.isfinite(systematic)]

            if valid.size > 0:

                valid = valid.astype(np.float64, copy = False)

                if not (maxima[i] > minima[i]):

                    result_sum[i, 0x0000] += np.sum(valid)
                    result_count[i, 0x0000] += valid.size

                else:

                    indices = np.clip(np.searchsorted(result_edges[i], valid, side = 'right') - 1, 0, n_bins - 1)

                    result_sum[i] += np.bincount(indices, weights = valid, minlength = n_bins)
                    result_count[i] += np.bincount(indices, weights = None, minlength = n_bins)

    ####################################################################################################################

    @staticmethod
    def compute_equal_sky_area_edges_and_stats(systematics: typing.Union[np.ndarray, typing.Callable], n_bins: int, temp_n_bins: typing.Optional[int] = None, show_progress_bar: bool = False) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        if n_bins <= 0:

            raise ValueError('`n_bins` must be > 0')

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(systematics)

        if generator_builder is None:

            raise ValueError('`systematics` must not be None')

        ################################################################################################################
        # PASS 1: COMPUTE GLOBAL STATISTICS                                                                            #
        ################################################################################################################

        dim = None
        counts = None

        sum1 = None
        sum2 = None

        minima = None
        maxima = None

        ################################################################################################################

        n_iters = 0

        generator = generator_builder()

        for vectors in tqdm.tqdm(generator(), total = None, disable = not show_progress_bar):

            dim, counts, sum1, sum2, minima, maxima = Decontamination_Abstract._accumulate_global_stats(
                vectors,
                dim,
                counts,
                sum1,
                sum2,
                minima,
                maxima
            )

            n_iters += 1

        ################################################################################################################

        if dim is None:

            raise ValueError('Empty dataset')

        if np.any(counts <= 0):

            raise ValueError('At least one systematic contains no finite value')

        ################################################################################################################

        means = np.divide(
            sum1,
            counts,
            out = np.full(dim, np.nan, dtype = np.float64),
            where = counts > 0
        )

        rmss = np.sqrt(np.divide(
            sum2,
            counts,
            out = np.full(dim, np.nan, dtype = np.float64),
            where = counts > 0
        ))

        vars = np.divide(
            sum2 - (sum1 * sum1) / counts,
            counts - 1,
            out = np.full(dim, np.nan, dtype = np.float64),
            where = counts > 1
        )

        ################################################################################################################

        stds = np.sqrt(np.maximum(0.0, vars))

        ################################################################################################################
        # PASS 2: BUILD TEMPORARY HISTOGRAMS                                                                           #
        ################################################################################################################

        if temp_n_bins is None:

            tmp_n_bins = np.empty(dim, dtype = np.int64)

            for i in range(dim):

                tmp_n_bins[i] = Decontamination_Abstract._compute_default_temp_n_bins(n_bins, int(counts[i]))

        else:

            if temp_n_bins >= n_bins:

                tmp_n_bins = np.full(dim, temp_n_bins, dtype = np.int64)

            else:

                raise ValueError('`temp_n_bins` must be >= `n_bins`')

        ################################################################################################################

        hits = [np.zeros(tmp_n_bins[i], dtype = np.float64) for i in range(dim)]

        ################################################################################################################

        generator = generator_builder()

        for vectors in tqdm.tqdm(generator(), total = n_iters, disable = not show_progress_bar):

            Decontamination_Abstract._accumulate_temporary_histograms(
                vectors,
                dim,
                tmp_n_bins,
                minima,
                maxima,
                hits
            )

        ################################################################################################################
        # BUILD FINAL EDGES                                                                                            #
        ################################################################################################################

        result_edges = np.empty((dim, n_bins + 1), dtype = np.float64)

        ################################################################################################################

        for i in range(dim):

            Decontamination_Abstract._build_equal_sky_area_edges(
                result_edges[i],
                hits[i],
                float(minima[i]),
                float(maxima[i]),
                n_bins
            )

        ################################################################################################################
        # PASS 3: COMPUTE EXACT BIN CENTERS                                                                            #
        ################################################################################################################

        result_sum = np.zeros((dim, n_bins), dtype = np.float64)
        result_count = np.zeros((dim, n_bins), dtype = np.int64)

        ################################################################################################################

        generator = generator_builder()

        for vectors in tqdm.tqdm(generator(), total = n_iters, disable = not show_progress_bar):

            Decontamination_Abstract._accumulate_bin_centers(
                vectors,
                dim,
                n_bins,
                minima,
                maxima,
                result_edges,
                result_sum,
                result_count
            )

        ################################################################################################################

        result_centers = np.divide(
            result_sum,
            result_count,
            out = np.full((dim, n_bins), np.nan, dtype = np.float64),
            where = result_count > 0
        )

        ################################################################################################################

        return (
            result_edges,
            result_centers,
            minima,
            maxima,
            means,
            rmss,
            stds,
            counts
        )

    ####################################################################################################################

    def _generate_catalog(self, density: np.ndarray, mult_factor: float = 20.0, seed: typing.Optional[int] = None) -> np.ndarray:

        catalog = np.empty(0, dtype = [('ra', np.float32), ('dec', np.float32)])

        generator = generator_number_density.Generator_NumberDensity(self._nside, self._footprint, nest = True, seed = seed)

        for lon, lat in tqdm.tqdm(generator.generate(density, mult_factor = mult_factor, n_max_per_batch = 10_000)):

            rows = np.empty(lon.shape[0], dtype = catalog.dtype)
            rows['ra'] = lon
            rows['dec'] = lat

            catalog = np.append(catalog, rows)

        return catalog

########################################################################################################################
