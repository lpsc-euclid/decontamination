# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from . import healpix_rand_ang, generator_abstract

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
    nest : bool, default: **True**
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering.
    lonlat : bool, default: **True**
        If **True**, assumes longitude and latitude in degrees, otherwise, co-latitude and longitude in radians.
    seed : typing.Optional[int], default: **None**
        Seed for random generators.
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True, lonlat: bool = True, seed: typing.Optional[int] = None):

        ################################################################################################################

        super().__init__(nside, footprint, nest = nest, lonlat = lonlat, seed = seed)

    ####################################################################################################################

    def generate(self, mean_density: float = 10.0, n_max_per_batch: typing.Optional[int] = None) -> typing.Generator[typing.Tuple[np.ndarray, np.ndarray], None, None]:

        """
        Iterator that yields uniform galaxy positions.

        Parameters
        ----------
        mean_density : float, default: **10.0**
            Mean number of galaxies per HEALPix pixel.
        n_max_per_batch : typing.Optional[int], default: **None**
            Maximum number of galaxy positions to yield in one batch.

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy catalog (longitudes and latitudes) in degrees.
        """

        ################################################################################################################

        rng = np.random.default_rng(seed = self._seed)

        ################################################################################################################

        galaxies_per_pixels = rng.poisson(lam = mean_density, size = self._footprint.shape[0])

        ################################################################################################################

        for central_pixels in self._iterator(galaxies_per_pixels, n_max_per_batch):

            yield healpix_rand_ang(
                self._nside,
                central_pixels,
                lonlat = self._lonlat,
                rng = rng
            )

########################################################################################################################
