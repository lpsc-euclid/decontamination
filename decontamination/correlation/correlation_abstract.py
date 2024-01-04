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

        return None, None, None

########################################################################################################################
