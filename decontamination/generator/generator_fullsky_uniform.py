# -*- coding: utf-8 -*-
########################################################################################################################

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
    seed : typing.Optional[int]
        Seed for random generators (default: **None**).
    """

    ####################################################################################################################

    def __init__(self, seed: typing.Optional[int] = None):

        self._random_generator = np.random.default_rng(seed = seed)

    ####################################################################################################################

    def generate(self, number_density_map: typing.Union[float, np.ndarray], n_max_per_batch: typing.Optional[int] = None) -> typing.Generator[typing.Tuple[np.ndarray, np.ndarray], None, None]:

        """
        Iterator that yields uniform galaxy positions.

        Parameters
        ----------
        n_max_per_batch : typing.Optional[int]
            Maximum number of galaxy positions to yield in one batch (default: **None**).

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy position (longitudes and latitudes).
        """

        ################################################################################################################

        shape = np.shape(number_density_map)

        number_density_map = np.broadcast_to(number_density_map, shape)

        if len(shape) > 1:

            raise ValueError('Number density map is not a float or vector.')

        ################################################################################################################

        galaxies_per_pixels = self._random_generator.poisson(number_density_map)

        ################################################################################################################

        for s, e in batch_iterator(galaxies_per_pixels.shape[0], n_max_per_batch):

            ############################################################################################################

            lon = self._random_generator.uniform(-180.0, +180.0, size = galaxies_per_pixels[s: e])

            lat = np.rad2deg(np.arcsin(self._random_generator.uniform(-1.0, +1.0, size = galaxies_per_pixels[s: e])))

            ############################################################################################################

            yield lon, lat

########################################################################################################################
