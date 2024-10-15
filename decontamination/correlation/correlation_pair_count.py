# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import typing

import numpy as np

from . import correlation_abstract

########################################################################################################################

try:

    import treecorr

except ImportError:

    treecorr = None

########################################################################################################################

# noinspection PyPep8Naming, PyTypeChecker, DuplicatedCode
class Correlation_PairCount(correlation_abstract.Correlation_Abstract):

    """
    Galaxy-galaxy 2pt correlation functions (2PCF) using the TreeCorr library. Supports pair counting (≡ NN
    correlations).

    Parameters
    ----------
    data_lon : np.ndarray
        Galaxy catalog longitudes (in degrees).
    data_lat : np.ndarray
        Galaxy catalog latitudes (in degrees).
    min_sep : float
        Minimum galaxy separation being considered (in arcmins).
    max_sep : float
        Maximum galaxy separation being considered (in arcmins).
    n_bins : int
        Number of angular bins.
    bin_slop : float = **None**
        Optional precision parameter (see `TreeCorr documentation <https://rmjarvis.github.io/TreeCorr/_build/html/binning.html#bin-slop>`_).
    n_threads : int, default: **None** ≡ the number of cpu cores
        Optional number of OpenMP threads to use during the calculation.
    """

    ####################################################################################################################

    def __init__(self, data_lon: np.ndarray, data_lat: np.ndarray, min_sep: float, max_sep: float, n_bins: int, bin_slop: typing.Optional[float] = None, n_threads: typing.Optional[int] = None, random_lon: typing.Optional[np.ndarray] = None, random_lat: typing.Optional[np.ndarray] = None, data_w: typing.Optional[np.ndarray] = None, random_w: typing.Optional[np.ndarray] = None):

        ################################################################################################################

        if treecorr is None:

            raise ImportError('TreeCorr is not installed.')

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        self._bin_slop = bin_slop
        self._n_threads = n_threads

        ################################################################################################################

        self._data_catalog = self._build_catalog(data_lon, data_lat, data_w)

        self._random_catalog = self._build_catalog(random_lon, random_lat, random_w) \
                                        if random_lon is not None and random_lat is not None else self._data_catalog

    ####################################################################################################################

    @property
    def bin_slop(self) -> typing.Optional[float]:

        """Precision parameter for binning (see `TreeCorr documentation <https://rmjarvis.github.io/TreeCorr/_build/html/binning.html#bin-slop>`_)."""

        return self._bin_slop

    ####################################################################################################################

    @property
    def n_threads(self) -> typing.Optional[int]:

        """How many OpenMP threads to use during the calculation."""

        return self._n_threads

    ####################################################################################################################

    @staticmethod
    def _build_catalog(lon: np.ndarray, lat: np.ndarray, w: np.ndarray) -> 'treecorr.Catalog':

        return treecorr.Catalog(
            ra = lon,
            dec = lat,
            k = None,
            w = w,
            ra_units = 'degrees',
            dec_units = 'degrees',
        )

    ####################################################################################################################

    def calculate(self, estimator: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        if estimator == 'dd':
            return self._2pcf(self._data_catalog, self._data_catalog)
        if estimator == 'rr':
            return self._2pcf(self._random_catalog, self._random_catalog)
        if estimator == 'dr':
            return self._2pcf(self._data_catalog, self._random_catalog)
        if estimator == 'rd':
            return self._2pcf(self._random_catalog, self._data_catalog)

        else:

            if estimator == 'peebles_hauser':
                return self._estimator(False, False)
            if estimator == 'landy_szalay_1':
                return self._estimator(True, False)
            if estimator == 'landy_szalay_2':
                return self._estimator(True, True)

        ################################################################################################################

        raise ValueError('Invalid estimator (`dd`, `rr`, `dr`, `rd`, `peebles_hauser`, `landy_szalay1`, `landy_szalay_2` are authorized)')

    ####################################################################################################################

    def _process(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> 'treecorr.NNCorrelation':

        nn = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, bin_slop = self._bin_slop, sep_units = 'arcmin')

        nn.process(catalog1, catalog2, num_threads = self._n_threads)

        return nn

    ####################################################################################################################

    def _2pcf(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        nn = self._process(catalog1, catalog2)

        return np.exp(nn.meanlogr), nn.xi, np.sqrt(nn.varxi)

    ####################################################################################################################

    def _estimator(self, with_dr: bool, with_rd: bool) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        dd = self._process(self._data_catalog)
        rr = self._process(self._random_catalog)

        dr = self._process(self._data_catalog, self._random_catalog) if with_dr else None
        rd = self._process(self._random_catalog, self._data_catalog) if with_rd else None

        ################################################################################################################

        xi, varxi = dd.calculateXi(rr = rr, dr = dr, rd = rd)

        ################################################################################################################

        return np.exp(dd.meanlogr), xi, np.sqrt(varxi)

########################################################################################################################
