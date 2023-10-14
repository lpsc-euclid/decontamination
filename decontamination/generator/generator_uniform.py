# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from . import generator_from_density

########################################################################################################################

# noinspection PyPep8Naming
class Generator_Uniform(generator_from_density.Generator_FromDensity):

    """
    Uniform galaxy catalog generator.

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    footprint : np.ndarray
        HEALPix indices of the region where galaxies are generated.
    nest : bool
        If True, assumes NESTED pixel ordering, otherwise, RING pixel ordering (default: **True**).
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True):

        ################################################################################################################

        super().__init__(nside, footprint, nest)

    ####################################################################################################################

    def generate(self, mult_factor: float = 10.0, seed: typing.Optional[int] = None) -> typing.Tuple[np.ndarray, np.ndarray]:

        """
        Generates uniform galaxy positions.

        Parameters
        ----------
        mult_factor : float
            Mean number of galaxies per HEALPix pixel (default: **10.0**).
        seed : typing.Optional[int]
            Seed for *poisson* and *uniform* generators (default: **None**).

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy catalog (longitudes and latitudes).
        """

        ################################################################################################################

        density_map = np.ones(self._x_diamonds.shape, dtype = np.float32)

        return super().generate(density_map, mult_factor = mult_factor, seed = seed)

########################################################################################################################
