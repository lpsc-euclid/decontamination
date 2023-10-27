# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from ..algo import batch_iterator

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

    def generate(self, number_density_map: typing.Optional[np.ndarray], mult_factor: float = 10.0, n_max_per_batch: typing.Optional[int] = None) -> typing.Generator[typing.Tuple[np.ndarray, np.ndarray], None, None]:

        """
        Iterator that yields galaxy positions from a density map.

        Parameters
        ----------
        number_density_map : typing.Optional[np.ndarray]
            Number of galaxies per HEALPix pixels.
        mult_factor : float
            Statistics multiplication factor (default: **10.0**).
        n_max_per_batch : typing.Optional[int]
            Maximum number of galaxy positions to yield in one batch.

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy catalog (longitudes and latitudes).
        """

        if self._footprint.shape != number_density_map.shape:

            raise Exception('Inconsistent footprint size and number density map size.')

        ################################################################################################################

        galaxies_per_pixels = self._random_generator.poisson(number_density_map * mult_factor)

        np.clip(galaxies_per_pixels, 0.0, None, out = galaxies_per_pixels)

        ################################################################################################################

        if n_max_per_batch is None:

            n_max_per_batch = self._footprint.shape[0]

        ################################################################################################################

        for s, e in batch_iterator(self._footprint.shape[0], n_max_per_batch):

            ############################################################################################################

            batched_footprint = np.repeat(self._footprint[s: e], galaxies_per_pixels[s: e])

            ############################################################################################################

            yield healpix_rand_ang(
                self._nside,
                batched_footprint,
                lonlat = self._lonlat,
                rng = self._random_generator
            )

    ########################################################################################################################
