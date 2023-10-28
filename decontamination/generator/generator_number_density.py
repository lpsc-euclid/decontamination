# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from . import healpix_rand_ang, generator_abstract

########################################################################################################################

# noinspection PyPep8Naming
class Generator_NumberDensity(generator_abstract.Generator_Abstract):

    """
    Galaxy catalog generator from a density map.

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

    def generate(self, number_density_map: np.ndarray, mult_factor: typing.Optional[float] = 10.0, n_max_per_batch: typing.Optional[int] = None) -> typing.Generator[typing.Tuple[np.ndarray, np.ndarray], None, None]:

        """
        Iterator that yields galaxy positions from a density map.

        Parameters
        ----------
        number_density_map : np.ndarray
            Number of galaxies per HEALPix pixels.
        mult_factor : typing.Optional[float]
            Statistics multiplication factor (default: **10.0**).
        n_max_per_batch : typing.Optional[int]
            Maximum number of galaxy positions to yield in one batch (default: **None**).

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy catalog (longitudes and latitudes).
        """

        if self._footprint.shape != number_density_map.shape:

            raise Exception('Inconsistent footprint shape and number density map shape.')

        ################################################################################################################

        rng = np.random.default_rng(seed = self._seed)

        ################################################################################################################

        galaxies_per_pixels = rng.poisson(number_density_map * mult_factor)

        np.clip(galaxies_per_pixels, 0.0, None, out = galaxies_per_pixels)

        ################################################################################################################

        for batched_footprint in self.iterator(galaxies_per_pixels, n_max_per_batch):

            yield healpix_rand_ang(
                self._nside,
                batched_footprint,
                lonlat = self._lonlat,
                rng = rng
            )

########################################################################################################################
