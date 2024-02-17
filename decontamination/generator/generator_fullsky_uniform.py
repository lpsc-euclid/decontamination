# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
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
    seed : int, default: **None**
        Optional seed for random generators.
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
        mean_density : float, default: **10.0**
            Mean number of galaxies per HEALPix pixel.
        n_max_per_batch : int, default: **None**
            Optional maximum number of galaxy positions to yield in one batch.

        Returns
        -------
        np.ndarray
            Galaxy catalog longitudes (in degrees).
        np.ndarray
            Galaxy catalog latitudes (in degrees).
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
