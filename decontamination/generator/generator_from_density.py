# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from . import rand_ang, generator_abstract

########################################################################################################################

# noinspection PyPep8Naming
class Generator_FromDensity(generator_abstract.Generator_Abstract):

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
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, nest: bool = True):

        ################################################################################################################

        super().__init__(nside, footprint, nest)

    ####################################################################################################################

    def generate(self, density_map: typing.Optional[np.ndarray], mult_factor: float = 10.0, seed: typing.Optional[int] = None) -> typing.Tuple[np.ndarray, np.ndarray]:

        """
        Generates galaxy positions from a density map.

        Parameters
        ----------
        density_map : typing.Optional[np.ndarray]
            Number of galaxies per HEALPix pixels.
        mult_factor : float
            Statistics multiplication factor (default: **10.0**).
        seed : typing.Optional[int]
            Seed for *poisson* and *uniform* generators (default: **None**).

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            Galaxy catalog (longitudes and latitudes).
        """

        ################################################################################################################

        rng = np.random.default_rng(seed = seed)

        ################################################################################################################

        # TODO #

        ################################################################################################################

        return None, None

########################################################################################################################
