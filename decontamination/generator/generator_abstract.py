# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

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
        HEALPix indices of the region where galaxies are generated.
    nest : bool
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering (default: **True**).
    lonlat : bool
        If **True**, assumes longitude and latitude in degree, otherwise, co-latitude and longitude in radians (default: **True**).
    seed : typing.Optional[int]
        Seed for random generators (default: **None**).
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True, lonlat: bool = True, seed: typing.Optional[int] = None):

        ################################################################################################################

        self._nside = nside

        self._footprint = footprint

        ################################################################################################################

        self._nest = nest

        self._lonlat = lonlat

        ################################################################################################################

        self._random_generator = np.random.default_rng(seed = seed)

########################################################################################################################
