# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

########################################################################################################################

try:

    import treecorr

except ImportError:

    treecorr = None

########################################################################################################################

# noinspection PyPep8Naming, PyTypeChecker, DuplicatedCode
class Correlation_PairCount(object):

    """
    Pair count two-point angular correlation function.

    Parameters
    ----------
    catalog_lon : np.ndarray
        Galaxy catalog longitudes (in degrees).
    catalog_lat : np.ndarray
        Galaxy catalog latitudes (in degrees).
    min_sep : float
        Minimum separation being considered (in degrees).
    max_sep : float
        Maximum separation being considered (in degrees).
    n_bins : int
        Number of angular bins.
    """

    ####################################################################################################################

    def __init__(self, catalog_lon: np.ndarray, catalog_lat: np.ndarray, min_sep: float, max_sep: float, n_bins: int):

        ################################################################################################################

        if treecorr is None:

            raise ImportError('TreeCorr is not installed.')

        ################################################################################################################

        self._min_sep = min_sep
        self._max_sep = max_sep
        self._n_bins = n_bins

        ################################################################################################################

        self._tc_galaxy_catalog = treecorr.Catalog(
            ra = catalog_lon,
            dec = catalog_lat,
            ra_units = 'degrees',
            dec_units = 'degrees'
        )

        ################################################################################################################

        self._dd = treecorr.NNCorrelation(min_sep = min_sep, max_sep = max_sep, nbins = n_bins, sep_units = 'arcmin')
        self._dd.process(self._tc_galaxy_catalog)

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

    def calculate(self, estimator: str, random_lon: typing.Optional[np.ndarray] = None, random_lat: typing.Optional[np.ndarray] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Calculates the angular correlation function with the specified estimator.

        Peebles & Hauser estimator (1974):

        .. math::
            \\hat{\\xi}=\\frac{DD-RR}{RR}

        1st Landy & Szalay estimator (1993):

        .. math::
            \\hat{\\xi}=\\frac{DD-2DR-RR}{RR}

        2nd Landy & Szalay estimator (1993):

        .. math::
            \\hat{\\xi}=\\frac{DD-DR-RD-RR}{RR}

        Parameters
        ----------
        estimator : str
            Estimator being considered ("dd", "peebles_hauser", "landy_szalay_1", "landy_szalay_2").
        random_lon : np.ndarray, default: None
            Random catalog longitudes (in degrees).
        random_lat : np.ndarray, default: None
            Random catalog latitudes (in degrees).

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]
            The bin of angles, the angular correlations and the correlation errors: :math:`(\\theta,w(\\theta),w_\\text{err}(\\theta))`.
        """

        ################################################################################################################

        if estimator == 'dd':

            return self._calculate_dd()

        ################################################################################################################

        if random_lon is None\
           or                \
           random_lat is None:

            raise ValueError(f'Parameters `random_lon` and `random_lat` have be provided with estimator `{estimator}`.')

        ################################################################################################################

        tc_random_catalog = treecorr.Catalog(
            ra = random_lon,
            dec = random_lat,
            ra_units = 'degrees',
            dec_units = 'degrees'
        )

        ################################################################################################################

        if estimator == 'peebles_hauser':

            return self._calculate_peebles_hauser(tc_random_catalog)

        if estimator == 'landy_szalay_1':

            return self._calculate_landy_szalay_1(tc_random_catalog)

        if estimator == 'landy_szalay_2':

            return self._calculate_landy_szalay_2(tc_random_catalog)

        ################################################################################################################

        raise ValueError('Invalid estimator (`dd`, `peebles_hauser`, `landy_szalay1`, `landy_szalay_2`)')

    ####################################################################################################################

    def _calculate_dd(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        w_theta, w_theta_variance = self._dd.calculateXi(rr = None, dr = None, rd = None)

        w_theta_error = np.sqrt(w_theta_variance)

        theta = np.exp(self._dd.meanlogr)

        ################################################################################################################

        return theta, w_theta, w_theta_error

    ####################################################################################################################

    def _calculate_peebles_hauser(self, tc_random_catalog: treecorr.catalog.Catalog) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        rr = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        rr.process(tc_random_catalog)

        ################################################################################################################

        w_theta, w_theta_variance = self._dd.calculateXi(rr = rr, dr = None, rd = None)

        w_theta_error = np.sqrt(w_theta_variance)

        theta = np.exp(self._dd.meanlogr)

        ################################################################################################################

        return theta, w_theta, w_theta_error

    ####################################################################################################################

    def _calculate_landy_szalay_1(self, tc_random_catalog: treecorr.catalog.Catalog) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        rr = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        rr.process(tc_random_catalog)

        ################################################################################################################

        dr = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        dr.process(self._tc_galaxy_catalog, tc_random_catalog)

        ################################################################################################################

        w_theta, w_theta_variance = self._dd.calculateXi(rr = rr, dr = dr, rd = None)

        w_theta_error = np.sqrt(w_theta_variance)

        theta = np.exp(self._dd.meanlogr)

        ################################################################################################################

        return theta, w_theta, w_theta_error

    ####################################################################################################################

    def _calculate_landy_szalay_2(self, tc_random_catalog: treecorr.catalog.Catalog) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        rr = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        rr.process(tc_random_catalog)

        ################################################################################################################

        dr = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        dr.process(self._tc_galaxy_catalog, tc_random_catalog)

        ################################################################################################################

        rd = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        rd.process(tc_random_catalog, self._tc_galaxy_catalog)

        ################################################################################################################

        w_theta, w_theta_variance = self._dd.calculateXi(rr = rr, dr = dr, rd = rd)

        w_theta_error = np.sqrt(w_theta_variance)

        theta = np.exp(self._dd.meanlogr)

        ################################################################################################################

        return theta, w_theta, w_theta_error

########################################################################################################################
