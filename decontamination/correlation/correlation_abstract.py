# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import abc
import typing

import numpy as np
import healpy as hp

########################################################################################################################

# noinspection PyPep8Naming
class Correlation_Abstract(abc.ABC):

    """
    Abstract class for computing galaxy-galaxy 2-point correlation functions (2PCF).
    """

    ####################################################################################################################

    def __init__(self, min_sep: float, max_sep: float, n_bins: int):

        if min_sep == 0.0:

            min_sep = 0.1

        self._min_sep = min_sep
        self._max_sep = max_sep

        self._min_sep_rad = np.radians(min_sep / 60.0)
        self._max_sep_rad = np.radians(max_sep / 60.0)

        print(self._min_sep, self._max_sep, self._min_sep_rad, self._max_sep_rad)

        self._n_bins = n_bins

    ####################################################################################################################

    @property
    def min_sep(self) -> float:

        """Minimum separation (in arcmins)."""

        return self._min_sep

    ####################################################################################################################

    @property
    def max_sep(self) -> float:

        """Maximum separation (in arcmins)."""

        return self._max_sep

    ####################################################################################################################

    @property
    def n_bins(self) -> int:

        """ Number of angular bins."""

        return self._n_bins

    ####################################################################################################################

    @abc.abstractmethod
    def calculate(self, estimator: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Calculates the angular correlation function.

        Peebles & Hauser estimator (1974):

        .. math::
            \\hat{\\xi}=\\frac{DD}{RR}-1

        1st Landy & Szalay estimator (1993):

        .. math::
            \\hat{\\xi}=\\frac{DD-2DR-RR}{RR}

        2nd Landy & Szalay estimator (1993):

        .. math::
            \\hat{\\xi}=\\frac{DD-DR-RD-RR}{RR}

        Parameters
        ----------
        estimator : str
            Estimator being considered ("dd", "rr", "dr", "rd", "peebles_hauser", "landy_szalay_1", "landy_szalay_2").

        Returns
        -------
        np.ndarray
            The bin of angles :math:`\\theta` (in arcmins).
        np.ndarray
            The angular correlations function :math:`\\xi(\\theta)`.
        np.ndarray
            The angular correlation function errors :math:`\\sigma_\\xi(\\theta)`.
        """

        return (
            np.array(0, dtype = np.float32),
            np.array(0, dtype = np.float32),
            np.array(0, dtype = np.float32),
        )

    ####################################################################################################################

    @staticmethod
    def _build_full_sky_contrast(nside: int, nest: bool, footprint: np.ndarray, catalog_lon: np.ndarray, catalog_lat: np.ndarray) -> np.ndarray:

        ################################################################################################################

        galaxy_pixels = hp.ang2pix(nside, catalog_lon, catalog_lat, nest = nest, lonlat = True)

        result = np.zeros(hp.nside2npix(nside), dtype = np.float32)

        np.add.at(result, galaxy_pixels, 1.0)

        ################################################################################################################

        mean = np.mean(result[footprint])

        result[footprint] -= mean
        result[footprint] /= mean

        ################################################################################################################

        return result

########################################################################################################################
