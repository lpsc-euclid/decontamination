# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

import healpy as hp

########################################################################################################################

# noinspection PyPep8Naming
class Generator_Abstract(object):

    """
    Abstract galaxy catalog generator.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the region where galaxies must be generated.
    nest : bool, default: **True**
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering.
    lonlat : bool, default: **True**
        If **True**, assumes longitude and latitude in degrees, otherwise, co-latitude and longitude in radians.
    seed : int, default: **None**
        Optional seed for random generators.
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True, lonlat: bool = True, seed: typing.Optional[int] = None):

        ################################################################################################################

        self._seed = seed
        self._nside = nside
        self._lonlat = lonlat

        ################################################################################################################

        self._footprint = footprint if nest else hp.ring2nest(nside, footprint)

    ####################################################################################################################

    def _iterator(self, galaxies_per_pixels: np.ndarray, n_max_per_batch: typing.Optional[int]) -> typing.Generator[np.ndarray, None, None]:

        """Iterator for splitting galaxy generation into batches."""

        ################################################################################################################

        if n_max_per_batch is None:

            yield np.repeat(self._footprint, galaxies_per_pixels)

            return

        ################################################################################################################

        n_total = np.sum(galaxies_per_pixels)

        ################################################################################################################

        start = 0
        stop = 0

        size = 0x00
        step = 1000

        ################################################################################################################

        while n_total > 0:

            ############################################################################################################

            q = np.cumsum(galaxies_per_pixels[stop: stop + step])

            ############################################################################################################

            if size + q[-1] < min(n_total, n_max_per_batch):

                ########################################################################################################
                # BATCH NOT FULL                                                                                       #
                ########################################################################################################

                stop += step
                size += q[-1]

                ########################################################################################################

            else:

                ########################################################################################################
                # BATCH FULL                                                                                           #
                ########################################################################################################

                stop += np.searchsorted(q, n_max_per_batch - size, side = 'right')

                if stop == start:

                    stop += 1

                ########################################################################################################

                batched_footprint = np.repeat(self._footprint[start: stop], galaxies_per_pixels[start: stop])

                n_total -= batched_footprint.shape[0]
                start = stop
                size = 0x00

                ########################################################################################################

                yield batched_footprint

########################################################################################################################
