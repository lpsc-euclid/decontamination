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
    def _compute_same_area_edges_step2(result_edges, hist, minimum, maximum, n_bins):

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

                result_edges[idx] = ((j + used_proportion) / hist.shape[0]) * (maximum - minimum) + minimum

                idx += 1
                acc = excess

        ################################################################################################################

        result_edges[0x0000] = minimum
        result_edges[n_bins] = maximum

    ####################################################################################################################

    @staticmethod
    def compute_same_area_edges(systematics: typing.Union[np.ndarray, typing.Callable], n_bins: int, is_normalized: bool = True) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        dim = systematics.shape[0]

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(systematics)

        ################################################################################################################
        # RENORMALIZE                                                                                                  #
        ################################################################################################################

        if is_normalized:

            minima = np.full(dim, 0.0, dtype = np.float32)
            maxima = np.full(dim, 1.0, dtype = np.float32)

        else:

            minima = np.full(dim, +np.inf, dtype = np.float32)
            maxima = np.full(dim, -np.inf, dtype = np.float32)

            generator = generator_builder()

            for vectors in generator():

                for i in range(dim):

                    minimum = np.nanmin(vectors)
                    maximum = np.nanmax(vectors)

                    if minima[i] > minimum:
                        minima[i] = minimum

                    if maxima[i] < maximum:
                        maxima[i] = maximum

        ################################################################################################################
        # BUILD HISTOGRAMS                                                                                             #
        ################################################################################################################

        tmp_n_bins = 100

        ################################################################################################################

        hist = np.zeros((dim, tmp_n_bins), dtype = np.float32)

        ################################################################################################################

        generator = generator_builder()

        for vectors in generator():

            for i in range(dim):

                temp, _ = np.histogram(vectors[i, :], bins = tmp_n_bins, range = (minima[i], maxima[i]))

                hist += temp

        ################################################################################################################
        # REBIN HISTOGRAM                                                                                              #
        ################################################################################################################

        result = np.empty((dim, n_bins + 1), dtype = np.float32)

        ################################################################################################################

        for i in range(dim):

            Decontamination_Abstract._compute_same_area_edges_step2(
                result[i],
                hist[i],
                minima[i],
                maxima[i],
                n_bins
            )

        ################################################################################################################

        return result, minima, maxima

########################################################################################################################
