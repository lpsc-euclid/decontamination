# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import numba as nb

from ..algo import dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class Decontamination_Abstract(object):

    """
    Systematics decontamination (abstract class).
    """

    ####################################################################################################################

    @staticmethod
    @nb.njit(fastmath = True)
    def _compute_same_sky_area_edges_step2(result_edges, result_centers, hits, vals, minimum, maximum, n_bins):

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

            if acc_n >= area:

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
    def compute_same_sky_area_edges_and_stats(systematics: typing.Union[np.ndarray, typing.Callable], n_bins: int, temp_n_bins: typing.Optional[float] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        dim = systematics.shape[0]

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(systematics)

        ################################################################################################################
        # RENORMALIZE                                                                                                  #
        ################################################################################################################

        n_vectors = 0

        sum1 = np.full(dim, 0.0, dtype = np.float32)
        sum2 = np.full(dim, 0.0, dtype = np.float32)

        minima = np.full(dim, +np.inf, dtype = np.float32)
        maxima = np.full(dim, -np.inf, dtype = np.float32)

        ################################################################################################################

        generator = generator_builder()

        for vectors in generator():

            n_vectors += vectors.shape[1]

            for i in range(dim):

                sum1[i] += np.sum(vectors ** 1, axis = 1)
                sum2[i] += np.sum(vectors ** 2, axis = 1)

                minimum = np.nanmin(vectors, axis = 1)
                maximum = np.nanmax(vectors, axis = 1)

                if minima[i] > minimum:
                    minima[i] = minimum

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

        for vectors in generator():

            for i in range(dim):

                temp, _ = np.histogram(vectors[i, :], bins = tmp_n_bins[i], range = (minima[i], maxima[i]))
                hits += temp

                temp, _ = np.histogram(vectors[i, :], bins = tmp_n_bins[i], range = (minima[i], maxima[i]), weights = vectors[i, :])
                vals += temp

        ################################################################################################################
        # REBIN HISTOGRAMS                                                                                             #
        ################################################################################################################

        result_edges = np.empty((dim, n_bins + 1), dtype = np.float32)
        result_centers = np.empty((dim, n_bins + 1), dtype = np.float32)

        ################################################################################################################

        for i in range(dim):

            Decontamination_Abstract._compute_same_sky_area_edges_step2(
                result_edges[i],
                result_centers[i],
                hits[i],
                vals[i],
                minima[i],
                maxima[i],
                n_bins
            )

        ################################################################################################################

        return result_edges, result_centers, minima, maxima, means, rmss, stds

########################################################################################################################
