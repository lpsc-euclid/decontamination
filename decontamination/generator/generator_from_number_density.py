# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from . import rand_ang, generator_abstract

########################################################################################################################

# noinspection PyPep8Naming
class Generator_FromNumberDensity(generator_abstract.Generator_Abstract):

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

    def generate(self, number_density_map: typing.Optional[np.ndarray], mult_factor: float = 10.0) -> typing.Tuple[np.ndarray, np.ndarray]:

        """
        Generates galaxy positions from a density map.

        Parameters
        ----------
        number_density_map : typing.Optional[np.ndarray]
            Number of galaxies per HEALPix pixels.
        mult_factor : float
            Statistics multiplication factor (default: **10.0**).

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy catalog (longitudes and latitudes).
        """

        ################################################################################################################

        # TODO #

        ################################################################################################################

        return None, None

########################################################################################################################
