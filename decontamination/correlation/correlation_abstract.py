# -*- coding: utf-8 -*-
########################################################################################################################

import abc
import typing

import numpy as np

########################################################################################################################

# noinspection PyPep8Naming
class Correlation_Abstract(abc.ABC):

    ####################################################################################################################

    def __init__(self, min_sep: float, max_sep: float, n_bins: int):

        self._min_sep = min_sep
        self._max_sep = max_sep
        self._n_bins = n_bins

    ####################################################################################################################

    @property
    def min_sep(self) -> float:

        """Minimum separation (in degrees)."""

        return self._min_sep

    ####################################################################################################################

    @property
    def max_sep(self) -> float:

        """Maximum separation (in degrees)."""

        return self._max_sep

    ####################################################################################################################

    @property
    def n_bins(self) -> int:

        """ Number of angular bins."""

        return self._n_bins

    ####################################################################################################################

    @abc.abstractmethod
    def calculate(self, estimator: str, random_lon: typing.Optional[np.ndarray] = None, random_lat: typing.Optional[np.ndarray] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

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
        random_lon : np.ndarray, default: None
            Random catalog longitudes (in degrees). For Peebles & Hauser and Landy & Szalay estimators only.
        random_lat : np.ndarray, default: None
            Random catalog latitudes (in degrees). For Peebles & Hauser and Landy & Szalay estimators only.

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray]
            The bin of angles :math:`\\theta` (in arcmins), the angular correlations :math:`\\xi(\\theta)` and the angular correlation errors :math:`\sigma_\\xi(\\theta)`.
        """

        return (
            np.array(0, dtype = np.float32),
            np.array(0, dtype = np.float32),
            np.array(0, dtype = np.float32),
        )

########################################################################################################################
