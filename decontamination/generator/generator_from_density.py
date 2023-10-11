# -*- coding: utf-8 -*-
########################################################################################################################

import math
import typing

import numpy as np

from . import xy2thetaphi, get_cell_size, generator_abstract

########################################################################################################################

# noinspection PyPep8Naming
class Generator_FromDensity(generator_abstract.Generator_Abstract):

    """
    Galaxy catalog generator from a galaxy density map.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix pixel indices of the footprint.
    nest : bool
        Tf True, assumes NESTED pixel ordering, otherwise, RING pixel ordering (default: **True**).
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True):

        ################################################################################################################

        super().__init__(nside, footprint, nest)

    ####################################################################################################################

    def generate(self, density_map: np.ndarray, mult_factor: float = 10.0, seed: typing.Optional[int] = None) -> typing.Tuple[np.ndarray, np.ndarray]:

        """
        Parameters
        ----------
        density_map : np.ndarray
            Number of galaxies per HEALPix pixels.
        mult_factor : float
            Statistics nultiplication factor (default: **10.0**).
        seed : typing.Optional[int]
            Seed for *poisson* and *uniform* generators (default: **None**).

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy catalog (longitudes and latitudes).
        """

        ################################################################################################################

        if self._x_diamonds.size != density_map.size\
           or                                       \
           self._y_diamonds.size != density_map.size:

            raise Exception('Inconsistent number of pixels and weights')

        ################################################################################################################

        rng = np.random.default_rng(seed = seed)

        ################################################################################################################

        return Generator_FromDensity._generate(
            rng,
            self._nside,
            self._x_diamonds,
            self._y_diamonds,
            density_map,
            mult_factor
        )

    ####################################################################################################################

    @staticmethod
    def _generate(rng: np.random.Generator, nside: int, x_diamonds: np.ndarray, y_diamonds: np.ndarray, density_map: np.ndarray, mult_factor: float) -> typing.Tuple[np.ndarray, np.ndarray]:

        ################################################################################################################
        #                                                                                                              #
        ################################################################################################################

        n_galaxies_per_pixels = np.empty_like(density_map, dtype = np.int32)

        ################################################################################################################

        n_total_galaxies = 0

        for i in range(density_map.shape[0]):

            n_galaxies = rng.poisson(mult_factor * density_map[i])

            n_galaxies_per_pixels[i] = n_galaxies

            n_total_galaxies += n_galaxies

        ################################################################################################################
        #                                                                                                              #
        ################################################################################################################

        # HEALPix diamonds to squares -> +45° rotation.

        x_center = (x_diamonds - y_diamonds) / math.sqrt(2)
        y_center = (x_diamonds + y_diamonds) / math.sqrt(2)

        ################################################################################################################

        start_idx = 0

        cell_size = get_cell_size(nside)

        x_galaxies = np.empty(n_total_galaxies, dtype = np.float32)
        y_galaxies = np.empty(n_total_galaxies, dtype = np.float32)

        for i, n_galaxies in enumerate(n_galaxies_per_pixels):

            end_idx = start_idx + n_galaxies

            dx, dy = rng.uniform(-0.5, +0.5, size = (2, n_galaxies))

            x_galaxies[start_idx: end_idx] = x_center[i] + dx * cell_size
            y_galaxies[start_idx: end_idx] = y_center[i] + dy * cell_size

            start_idx = end_idx

        ################################################################################################################

        # Squares to HEALPix diamonds -> -45° rotation.

        x_galaxies2 = (+x_galaxies + y_galaxies) / math.sqrt(2)
        y_galaxies2 = (-x_galaxies + y_galaxies) / math.sqrt(2)

        ################################################################################################################
        #                                                                                                              #
        ################################################################################################################

        theta, phi, = xy2thetaphi(x_galaxies2, y_galaxies2)

        lon = 00.0 + 180.0 * phi / np.pi
        lat = 90.0 - 180.0 * theta / np.pi

        ################################################################################################################

        return lon, lat

########################################################################################################################
