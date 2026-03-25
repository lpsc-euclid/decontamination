# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import tqdm
import typing

import numpy as np

from ..generator import generator_number_density

########################################################################################################################

from . import compute_equal_area_binning_and_statistics as _compute_binning
from . import compute_equal_area_correlation as _compute_correlation

########################################################################################################################

# noinspection PyPep8Naming
class Decontamination_Abstract(object):

    """
    Systematics decontamination (abstract class).
    """

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, coverage: np.ndarray, footprint_systematics: typing.Union[np.ndarray, typing.Callable], galaxy_number_density: np.ndarray):

        ################################################################################################################

        self._nside = nside
        self._footprint = footprint
        self._coverage = coverage
        self._footprint_systematics = footprint_systematics
        self._galaxy_number_density = galaxy_number_density

        ################################################################################################################

        self._corrected_galaxy_number_density = galaxy_number_density / coverage

    ####################################################################################################################

    @property
    def nside(self) -> int:

        """Nside."""

        return self._nside

    ####################################################################################################################

    @property
    def footprint(self) -> np.ndarray:

        """Footprint."""

        return self._footprint

    ####################################################################################################################

    @property
    def coverage(self) -> np.ndarray:

        """Coverage."""

        return self._coverage

    ####################################################################################################################

    @property
    def footprint_systematics(self) -> typing.Union[np.ndarray, typing.Callable]:

        """Footprint systematics."""

        return self._footprint_systematics

    ####################################################################################################################

    @property
    def galaxy_number_density(self) -> np.ndarray:

        """Galaxy number density."""

        return self._galaxy_number_density

    ####################################################################################################################

    @property
    def corrected_galaxy_number_density(self) -> np.ndarray:

        """Coverage-corrected galaxy number density."""

        return self._corrected_galaxy_number_density

    ####################################################################################################################

    def compute_equal_area_binning_and_statistics(self, n_bins: int, temp_n_bins: typing.Optional[int] = None, exact: bool = False, show_progress_bar: bool = False) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:

        return _compute_binning(self._footprint_systematics, n_bins, temp_n_bins, exact, show_progress_bar)

    ####################################################################################################################

    def compute_equal_area_correlation(self, galaxy_number_density: np.ndarray, edges: np.ndarray) -> np.ndarray:

        return _compute_correlation(self._footprint_systematics, galaxy_number_density, edges)

    ####################################################################################################################

    def _generate_catalog(self, density: np.ndarray, mult_factor: float = 20.0, seed: typing.Optional[int] = None) -> np.ndarray:

        catalog = np.empty(0, dtype = [('ra', np.float32), ('dec', np.float32)])

        generator = generator_number_density.Generator_NumberDensity(self._nside, self._footprint, nest = True, seed = seed)

        for lon, lat in tqdm.tqdm(generator.generate(density, mult_factor = mult_factor, n_max_per_batch = 10_000)):

            rows = np.empty(lon.shape[0], dtype = catalog.dtype)
            rows['ra'] = lon
            rows['dec'] = lat

            catalog = np.append(catalog, rows)

        return catalog

########################################################################################################################
