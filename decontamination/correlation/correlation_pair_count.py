# -*- coding: utf-8 -*-
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
    Galaxy angular correlation function from pair counting.

    Parameters
    ----------
    catalog_lon : np.ndarray
        Galaxy catalog longitudes (in degrees).
    catalog_lat : np.ndarray
        Galaxy catalog latitudes (in degrees).
    min_sep : float
        Minimum galaxy separation being considered (in arcmins).
    max_sep : float
        Maximum galaxy separation being considered (in arcmins).
    n_bins : int
        Number of angular bins.
    """

    ####################################################################################################################

    def __init__(self, catalog_lon: np.ndarray, catalog_lat: np.ndarray, min_sep: float, max_sep: float, n_bins: int):

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        if treecorr is None:

            raise ImportError('TreeCorr is not installed.')

        ################################################################################################################

        self._data_catalog = treecorr.Catalog(
            ra = catalog_lon,
            dec = catalog_lat,
            ra_units = 'degrees',
            dec_units = 'degrees'
        )

        ################################################################################################################
        # CORRELATE                                                                                                    #
        ################################################################################################################

        self._dd = self._correlate(self._data_catalog, None)

    ####################################################################################################################

    def _correlate(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> 'treecorr.NNCorrelation':

        result = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')

        result.process(catalog1, catalog2)

        return result

    ####################################################################################################################

    def calculate(self, estimator: str, random_lon: typing.Optional[np.ndarray] = None, random_lat: typing.Optional[np.ndarray] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        if estimator == 'dd':

            return self._calculate_xy(self._data_catalog, None)

        ################################################################################################################

        if random_lon is None\
           or                \
           random_lat is None:

            raise ValueError(f'Parameters `random_lon` and `random_lat` have be provided with estimator `{estimator}`.')

        ################################################################################################################

        self_random_catalog = treecorr.Catalog(
            ra = random_lon,
            dec = random_lat,
            ra_units = 'degrees',
            dec_units = 'degrees'
        )

        ################################################################################################################

        if estimator == 'rr':
            return self._calculate_xy(self_random_catalog, None)
        if estimator == 'dr':
            return self._calculate_xy(self._data_catalog, self_random_catalog)
        if estimator == 'rd':
            return self._calculate_xy(self_random_catalog, self._data_catalog)
        if estimator == 'peebles_hauser':
            return self._calculate_xi(self_random_catalog, False, False)
        if estimator == 'landy_szalay_1':
            return self._calculate_xi(self_random_catalog, True, False)
        if estimator == 'landy_szalay_2':
            return self._calculate_xi(self_random_catalog, True, True)

        ################################################################################################################

        raise ValueError('Invalid estimator (`dd`, `rr`, `dr`, `rd`, `peebles_hauser`, `landy_szalay1`, `landy_szalay_2`)')

    ####################################################################################################################

    def _calculate_xy(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        theta = np.exp(self._dd.meanlogr)

        ################################################################################################################

        if catalog1 is self._data_catalog or catalog2 is not None:

            xy = self._correlate(catalog1, catalog2)

        else:

            xy = self._dd

        ################################################################################################################

        xi_theta = np.diff(xy.npairs, prepend = 0)

        xi_theta_error = np.sqrt(xy.npairs)

        ################################################################################################################

        return theta, xi_theta, xi_theta_error

    ####################################################################################################################

    def _calculate_xi(self, self_random_catalog: 'treecorr.Catalog', with_dr: bool, with_rd: bool) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        theta = np.exp(self._dd.meanlogr)

        ################################################################################################################

        xi_theta, xi_theta_variance = self._dd.calculateXi(
            rr = self._correlate(self_random_catalog),
            dr = self._correlate(self._data_catalog, self_random_catalog) if with_dr else None,
            rd = self._correlate(self_random_catalog, self._data_catalog) if with_rd else None,
        )

        xi_theta_error = np.sqrt(xi_theta_variance)

        ################################################################################################################

        return theta, xi_theta, xi_theta_error

########################################################################################################################
