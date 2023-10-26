# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from ..algo import batch_iterator

from . import rand_ang, generator_abstract

########################################################################################################################

# noinspection PyPep8Naming
class Generator_LogNormal(generator_abstract.Generator_Abstract):

    """
    Log-normal galaxy catalog generator.

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

    def generate(self, mean_density: float = 10.0, n_max_per_batch: typing.Optional[int] = None) -> typing.Generator[typing.Tuple[np.ndarray, np.ndarray], None, None]:

        """
        Iterator that yields log-normal galaxy positions.

        Parameters
        ----------
        mean_density : float
            Mean number of galaxies per HEALPix pixel (default: **10.0**).
        n_max_per_batch : typing.Optional[int]
            Maximum number of galaxy positions to yield in one batch.

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy catalog (longitudes and latitudes).
        """

        ################################################################################################################

        galaxies_per_pixels = self._random_generator.poisson(lam = mean_density, size = self._footprint.shape[0])

        ################################################################################################################

        if n_max_per_batch is None:

            n_max_per_batch = self._footprint.shape[0]

        ################################################################################################################

        for s, e in batch_iterator(self._footprint.shape[0], n_max_per_batch):

            ############################################################################################################

            batched_footprint = np.repeat(self._footprint[s: e], galaxies_per_pixels[s: e])

            ############################################################################################################

            yield rand_ang(
                self._nside,
                batched_footprint,
                lonlat = self._lonlat,
                rng = self._random_generator
            )

    ########################################################################################################################
