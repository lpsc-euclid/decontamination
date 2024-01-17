# -*- coding: utf-8 -*-
########################################################################################################################

import math
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
    def _compute_same_sky_area_edges_step2(result_edges, hist, minimum, maximum, n_bins):

        ################################################################################################################

        idx = 1
        acc = 0.0

        area = np.sum(hist) / n_bins

        ################################################################################################################

        for j in range(hist.shape[0]):

            val = hist[j]

            acc += val

            if acc >= area:

                excess = acc - area

                used_proportion = (val - excess) / val

                result_edges[idx] = ((j + used_proportion + 1) / hist.shape[0]) * (maximum - minimum) + minimum

                idx += 1
                acc = excess

        ################################################################################################################

        result_edges[0x0000] = minimum
        result_edges[n_bins] = maximum

    ####################################################################################################################

    @staticmethod
    def compute_same_sky_area_edges_and_stats(systematics: typing.Union[np.ndarray, typing.Callable], n_bins: int) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

            n_vectors += vectors.shape[0]

            for i in range(dim):

                sum1[i] += np.sum(vectors ** 1)
                sum2[i] += np.sum(vectors ** 2)

                minimum = np.nanmin(vectors)
                maximum = np.nanmax(vectors)

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

        tmp_n_bins = np.full(dim, int(1.0 + math.log2(n_vectors)), np.int64)  # Sturges' rule

        ################################################################################################################

        area = n_vectors / n_bins

        for i in range(dim):

            h_max = 0.68 * n_vectors / stds[i]

            while h_max / tmp_n_bins[i] > 2.0 * area:

                tmp_n_bins[i] *= 2

        ################################################################################################################
        # BUILD HISTOGRAMS                                                                                             #
        ################################################################################################################

        hist = [np.zeros(tmp_n_bins[i], dtype = np.float32) for i in range(dim)]

        ################################################################################################################

        generator = generator_builder()

        for vectors in generator():

            for i in range(dim):

                temp, _ = np.histogram(vectors[i, :], bins = tmp_n_bins[i], range = (minima[i], maxima[i]))

                hist += temp

        ################################################################################################################
        # REBIN HISTOGRAM                                                                                              #
        ################################################################################################################

        result = np.empty((dim, n_bins + 1), dtype = np.float32)

        ################################################################################################################

        for i in range(dim):

            Decontamination_Abstract._compute_same_sky_area_edges_step2(
                result[i],
                hist[i],
                minima[i],
                maxima[i],
                n_bins
            )

        ################################################################################################################

        return result, minima, maxima, means, rmss, stds

########################################################################################################################
