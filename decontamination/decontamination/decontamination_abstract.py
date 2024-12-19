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

from ..algo import dataset_to_generator_builder
from ..generator import generator_number_density

########################################################################################################################

# noinspection PyPep8Naming
class Decontamination_Abstract(object):

    """
    Systematics decontamination (abstract class).
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, coverage: np.ndarray, footprint_systematics: np.ndarray, galaxy_number_density: np.ndarray):

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
    @nb.njit(fastmath = True)
    def _compute_equal_sky_area_edges_step2(result_edges: np.ndarray, result_centers: np.ndarray, hits: np.ndarray, vals: np.ndarray, minimum: np.ndarray, maximum: np.ndarray, n_bins: int) -> None:

        ################################################################################################################

        idx = 0

        acc_n = 0.0
        acc_v = 0.0

        area = np.sum(hits) / n_bins

        ################################################################################################################

        for j in range(hits.shape[0]):

            cur_n = hits[j]
            cur_v = vals[j]

            if cur_n == 0:

                continue

            acc_n += cur_n
            acc_v += cur_v

            if acc_n >= area or j == hits.shape[0] - 1:

                ########################################################################################################

                excess_n = acc_n - area

                excess_v = cur_v * (excess_n / cur_n)

                ########################################################################################################

                result_edges[idx + 1] = ((j + (cur_n - excess_n) / cur_n) / hits.shape[0]) * (maximum - minimum) + minimum

                result_centers[idx + 0] = (acc_v - excess_v) / (acc_n - excess_n)

                ########################################################################################################

                idx += 1

                acc_n = excess_n
                acc_v = excess_v

        ################################################################################################################

        result_edges[0x0000] = minimum
        result_edges[n_bins] = maximum

    ####################################################################################################################

    @staticmethod
    def compute_equal_sky_area_edges_and_stats(systematics: typing.Union[np.ndarray, typing.Callable], n_bins: int, temp_n_bins: typing.Optional[float] = None, show_progress_bar: bool = False) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:

        ################################################################################################################

        dim = systematics.shape[0]

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(systematics)

        ################################################################################################################
        # COMPUTE STATISTICS                                                                                           #
        ################################################################################################################

        n_iters = 0
        n_vectors = 0

        sum1 = np.full(dim, 0.0, dtype = np.float32)
        sum2 = np.full(dim, 0.0, dtype = np.float32)

        minima = np.full(dim, +np.inf, dtype = np.float32)
        maxima = np.full(dim, -np.inf, dtype = np.float32)

        ################################################################################################################

        generator = generator_builder()

        for vectors in tqdm.tqdm(generator(), total = None, disable = not show_progress_bar):

            n_iters += 0x00000000000001
            n_vectors += vectors.shape[1]

            for i in range(dim):

                systematics = vectors[i]

                sum1[i] += np.nansum(systematics ** 1)
                sum2[i] += np.nansum(systematics ** 2)

                minimum = np.nanmin(systematics)

                if minima[i] > minimum:
                    minima[i] = minimum

                maximum = np.nanmax(systematics)

                if maxima[i] < maximum:
                    maxima[i] = maximum

        ################################################################################################################

        means = sum1 / n_vectors

        rmss = np.sqrt(sum2 / n_vectors)

        stds = np.sqrt((sum2 - (sum1 ** 2) / n_vectors) / (n_vectors - 1))

        ################################################################################################################
        # ESTIMATE BINNING                                                                                             #
        ################################################################################################################

        default_n_bins = np.int64(10.0 * n_bins * (1.0 + np.log2(n_vectors)))

        print('default_n_bins', default_n_bins)

        ################################################################################################################

        tmp_n_bins = np.full(dim, default_n_bins if temp_n_bins is None else temp_n_bins, np.int64)

        ################################################################################################################
        # BUILD HISTOGRAMS                                                                                             #
        ################################################################################################################

        hits = [np.zeros(tmp_n_bins[i], dtype = np.float32) for i in range(dim)]
        vals = [np.zeros(tmp_n_bins[i], dtype = np.float32) for i in range(dim)]

        ################################################################################################################

        generator = generator_builder()

        for vectors in tqdm.tqdm(generator(), total = n_iters, disable = not show_progress_bar):

            for i in range(dim):

                temp, _ = np.histogram(vectors[i, :], bins = tmp_n_bins[i], range = (minima[i], maxima[i]), weights = None)
                hits[i] += temp

                temp, _ = np.histogram(vectors[i, :], bins = tmp_n_bins[i], range = (minima[i], maxima[i]), weights = vectors[i, :])
                vals[i] += temp

        ################################################################################################################
        # REBIN HISTOGRAMS                                                                                             #
        ################################################################################################################

        result_edges = np.empty((dim, n_bins + 1), dtype = np.float32)
        result_centers = np.empty((dim, n_bins + 0), dtype = np.float32)

        ################################################################################################################

        for i in range(dim):

            Decontamination_Abstract._compute_equal_sky_area_edges_step2(
                result_edges[i],
                result_centers[i],
                hits[i],
                vals[i],
                minima[i],
                maxima[i],
                n_bins
            )

        ################################################################################################################

        return result_edges, result_centers, minima, maxima, means, rmss, stds, n_vectors

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
