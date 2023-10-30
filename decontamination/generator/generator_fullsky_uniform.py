# -*- coding: utf-8 -*-
########################################################################################################################

import math
import typing

import numpy as np

from ..algo import batch_iterator

########################################################################################################################

# noinspection PyPep8Naming
class Generator_FullSkyUniform(object):

    """
    Uniform galaxy catalog generator over the sphere.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    seed : typing.Optional[int]
        Seed for random generators (default: **None**).
    """

    ####################################################################################################################

    def __init__(self, nside: int, seed: typing.Optional[int] = None):

        self._seed = seed
        self._nside = nside

    ####################################################################################################################

    def generate(self, mean_density: float, n_max_per_batch: typing.Optional[int] = None) -> typing.Generator[typing.Tuple[np.ndarray, np.ndarray], None, None]:

        """
        Iterator that yields uniform galaxy positions.

        Parameters
        ----------
        mean_density : float
            Mean number of galaxies per HEALPix pixel (default: **10.0**).
        n_max_per_batch : typing.Optional[int]
            Maximum number of galaxy positions to yield in one batch (default: **None**).

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy position (longitudes and latitudes).
        """

        ################################################################################################################

        rng = np.random.default_rng(seed = self._seed)

        ################################################################################################################

        n_galaxies = rng.poisson(math.floor(mean_density * 12 * self._nside ** 2))

        ################################################################################################################

        for start, stop in batch_iterator(n_galaxies, n_max_per_batch):

            ############################################################################################################

            lon = rng.uniform(-180.0, +180.0, size = stop - start)

            lat = np.rad2deg(np.arcsin(rng.uniform(-1.0, +1.0, size = stop - start)))

            ############################################################################################################

            yield lon, lat

########################################################################################################################
