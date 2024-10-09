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
    Galaxy angular correlation function using the TreeCorr library. Supports pair counting (≡ NN correlations)
    or a scalar field approach (≡ KK correlations) if a footprint is provided.

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

    def __init__(self, data_lon: np.ndarray, data_lat: np.ndarray, min_sep: float, max_sep: float, n_bins: int, bin_slop: typing.Optional[float] = None, n_threads: typing.Optional[int] = None, random_lon: typing.Optional[np.ndarray] = None, random_lat: typing.Optional[np.ndarray] = None):

        ################################################################################################################

        if treecorr is None:

            raise ImportError('TreeCorr is not installed.')

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        self._bin_slop = bin_slop
        self._n_threads = n_threads

        ################################################################################################################

        self._data_catalog = self._build_catalog(data_lon, data_lat)

        if random_lon is not None and random_lat is not None:

            self._random_catalog = self._build_catalog(random_lon, random_lat)

        else:

            self._random_catalog = None

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
    def _build_catalog(lon: np.ndarray, lat: np.ndarray) -> 'treecorr.Catalog':

        return treecorr.Catalog(
            ra = lon,
            dec = lat,
            ra_units = 'degrees',
            dec_units = 'degrees'
        )

    ####################################################################################################################

    def _correlate(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> typing.Union['treecorr.NNCorrelation', 'treecorr.KKCorrelation']:

        ################################################################################################################

        if self._bin_slop is None:
            result = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        else:
            result = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, bin_slop = self._bin_slop, sep_units = 'arcmin')

        ################################################################################################################

        result.process(catalog1, catalog2, num_threads = self._n_threads)

        ################################################################################################################

        return result

    ####################################################################################################################

    def calculate(self, estimator: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        if estimator == 'dd':

            return self._calculate_xy(self._data_catalog, None)

        ################################################################################################################

        if self._random_catalog is None:

            raise ValueError(f'Random catalog have to be provided with estimator `{estimator}`')

        ################################################################################################################

        if estimator == 'rr':
            return self._calculate_xy(self._random_catalog, None)
        if estimator == 'dr':
            return self._calculate_xy(self._data_catalog, self._random_catalog)
        if estimator == 'rd':
            return self._calculate_xy(self._random_catalog, self._data_catalog)

        else:

            if estimator == 'peebles_hauser':
                return self._calculate_xi(False, False)
            if estimator == 'landy_szalay_1':
                return self._calculate_xi(True, False)
            if estimator == 'landy_szalay_2':
                return self._calculate_xi(True, True)

        ################################################################################################################

        raise ValueError('Invalid estimator (`dd`, `rr`, `dr`, `rd`, `peebles_hauser`, `landy_szalay1`, `landy_szalay_2` are authorized)')

    ####################################################################################################################

    def _calculate_xy(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        xy = self._correlate(catalog1, catalog2)

        ################################################################################################################

        theta = np.exp(xy.meanlogr)

        xi_theta = xy.xi

        xi_theta_error = np.sqrt(xy.varxi)

        ################################################################################################################

        return theta, xi_theta, xi_theta_error

    ####################################################################################################################

    def _calculate_xi(self, with_dr: bool, with_rd: bool) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        dd = self._correlate(self._data_catalog)
        rr = self._correlate(self._random_catalog)

        dr = self._correlate(self._data_catalog, self._random_catalog) if with_dr else None
        rd = self._correlate(self._random_catalog, self._data_catalog) if with_rd else None

        ################################################################################################################

        theta = np.exp(dd.meanlogr)

        ################################################################################################################

        xi_theta, xi_theta_variance = dd.calculateXi(rr = rr, dr = dr, rd = rd)

        xi_theta_error = np.sqrt(xi_theta_variance)

        ################################################################################################################

        return theta, xi_theta, xi_theta_error

########################################################################################################################
