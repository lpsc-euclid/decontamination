# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from . import rand_ang, generator_abstract

########################################################################################################################

# noinspection PyPep8Naming
class Generator_Uniform(generator_abstract.Generator_Abstract):

    """
    Uniform galaxy catalog generator.

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

        super().__init__(nside, footprint, nest = nest, lonlat = lonlat, seed = seed)

    ####################################################################################################################

    def generate(self, mult_factor: float = 10.0) -> typing.Tuple[np.ndarray, np.ndarray]:

        """
        Generates uniform galaxy positions.

        Parameters
        ----------
        mult_factor : float
            Mean number of galaxies per HEALPix pixel (default: **10.0**).

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy catalog (longitudes and latitudes).
        """

        ################################################################################################################

        n_galaxies_per_pixels = self._random_generator.poisson(lam = mult_factor, size = self._footprint.shape[0])

        ################################################################################################################

        pixels = np.repeat(self._footprint, n_galaxies_per_pixels)

        ################################################################################################################

        lon, lat = rand_ang(
            self._nside,
            pixels,
            lonlat = self._lonlat,
            rng = self._random_generator
        )

        ################################################################################################################

        return lon, lat

########################################################################################################################
