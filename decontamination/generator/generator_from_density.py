
# -*- coding: utf-8 -*-
########################################################################################################################

import math
import typing

import numpy as np
import numba as nb

from . import xy2thetaphi, get_cell_size, generator_abstract

########################################################################################################################

# noinspection PyPep8Naming
class Generator_FromDensity(generator_abstract.Generator_Abstract):

    """
    Galaxy generator from a galaxy density map.
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True):

        ################################################################################################################

        super().__init__(nside, footprint, nest)

    ####################################################################################################################

    def generate(self, density_map: np.ndarray, mult_factor: float = 1.0, seed: typing.Optional[int] = None) -> typing.Tuple[np.ndarray, np.ndarray]:

        """
        Parameters
        ----------
        density_map : np.ndarray
            ???
        mult_factor : float
            ??? (default: **1.0**)
        seed : typing.Optional[int]
            Seed for random generator (default: **None**).
        """

        if self._x_center_diamond.size != density_map.size\
           or                                             \
           self._y_center_diamond.size != density_map.size:

            raise Exception('Inconsistent number of pixels and weights')

        return Generator_FromDensity._generate(
            self._nside,
            self._x_center_diamond,
            self._y_center_diamond,
            density_map,
            mult_factor,
            seed
        )

    ####################################################################################################################

    @staticmethod
    @nb.njit(fastmath = True)
    def _generate(nside: int, x_center_diamond: np.ndarray, y_center_diamond: np.ndarray, density_map: np.ndarray, mult_factor: float, seed: typing.Optional[int]) -> typing.Tuple[np.ndarray, np.ndarray]:

        ################################################################################################################

        rng = np.random.default_rng(seed = seed)

        ################################################################################################################

        n_galaxies_per_pixels = rng.poisson(mult_factor * density_map)

        n_total_galaxies = np.sum(n_galaxies_per_pixels)

        ################################################################################################################

        # HEALPix diamonds to squares -> +45° rotation.

        x_center = (x_center_diamond - y_center_diamond) / math.sqrt(2)
        y_center = (x_center_diamond + y_center_diamond) / math.sqrt(2)

        ################################################################################################################

        start_idx = 0

        cell_size = get_cell_size(nside)

        x_galaxies = np.empty(n_total_galaxies, dtype = np.float32)
        y_galaxies = np.empty(n_total_galaxies, dtype = np.float32)

        for i, n_galaxies in enumerate(n_galaxies_per_pixels):

            end_idx = start_idx + n_galaxies

            dx, dy = rng.uniform(-0.5, +0.5, size = (n_galaxies, 2))

            x_galaxies[start_idx: end_idx] = x_center[i] + dx * cell_size
            y_galaxies[start_idx: end_idx] = y_center[i] + dy * cell_size

            start_idx = end_idx

        ################################################################################################################

        # squares to HEALPix diamonds -> -45° rotation.

        x_galaxies_diamonds = (+x_galaxies + y_galaxies) / math.sqrt(2)
        y_galaxies_diamonds = (-x_galaxies + y_galaxies) / math.sqrt(2)

        ################################################################################################################

        theta, phi, = xy2thetaphi(x_galaxies_diamonds, y_galaxies_diamonds)

        lon = 00.0 + 180.0 * phi / np.pi
        lat = 90.0 - 180.0 * theta / np.pi

        ################################################################################################################

        return lon, lat

########################################################################################################################
