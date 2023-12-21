# -*- coding: utf-8 -*-
########################################################################################################################

import typing
import treecorr

import numpy as np

########################################################################################################################

# noinspection PyPep8Naming, PyTypeChecker
class Correlation_CountCount(object):

    """
    ???

    Parameters
    ----------
    catalog_ra : np.ndarray
        Galaxy catalog longitudes.
    catalog_dec : np.ndarray
        Galaxy catalog latitudes.
    min_sep : float
        The minimum separation being considered in degrees.
    max_sep : float
        The maximum separation being considered in degrees.
    n_bins : int
        The number of bins in logr.
    """

    ####################################################################################################################

    def __init__(self, catalog_ra: np.ndarray, catalog_dec: np.ndarray, min_sep: float, max_sep: float, n_bins: int):

        ################################################################################################################

        self._min_sep = min_sep
        self._max_sep = max_sep
        self._n_bins = n_bins

        ################################################################################################################

        self._tc_galaxy_catalog = treecorr.Catalog(
            ra = catalog_ra,
            dec = catalog_dec,
            ra_units = 'degrees',
            dec_units = 'degrees'
        )

        ################################################################################################################

        self._dd = treecorr.NNCorrelation(min_sep = min_sep, max_sep = max_sep, nbins = n_bins, sep_units = 'arcmin')
        self._dd.process(self._tc_galaxy_catalog)

    ####################################################################################################################

    def calculate(self, random_ra: np.ndarray, random_dec: np.ndarray, estimator: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Calculates the galaxy angular correlations with the specified estimator.

        Peebles & Hauser estimator:

        .. math::
            \\hat{\\xi}=\\frac{DD}{RR}-1

        Landy & Szalay estimator:

        .. math::
            \\hat{\\xi}=\\frac{DD-2DR-RR}{RR}

        Parameters
        ----------
        random_ra : np.ndarray
            Random catalog longitudes.
        random_dec : np.ndarray
            Random catalog latitudes.
        estimator : str
            Estimator to use for calculating the correlation ("peebles_hauser", "landy_szalay").

        Returns
        -------
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]
            The bin of angles, the angular correlations and the correlation errors.
        """

        ################################################################################################################

        tc_random_catalog = treecorr.Catalog(
            ra = random_ra,
            dec = random_dec,
            ra_units = 'degrees',
            dec_units = 'degrees'
        )

        ################################################################################################################

        if estimator == 'peebles_hauser':

            return self._calculate_peebles_hauser(tc_random_catalog)

        if estimator == 'landy_szalay':

            return self._calculate_landy_szalay(tc_random_catalog)

        ################################################################################################################

        raise ValueError('Invalid estimator (`peebles_hauser`, `landy_szalay`)')

    ####################################################################################################################

    def _calculate_peebles_hauser(self, tc_random_catalog: treecorr.catalog.Catalog) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        rr = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        rr.process(tc_random_catalog)

        ################################################################################################################

        wtheta, wtheta_variance = self._dd.calculateXi(rr = rr, dr = None, rd = None)

        wtheta_error = np.sqrt(wtheta_variance)

        theta = np.exp(self._dd.meanlogr)

        ################################################################################################################

        return theta, wtheta, wtheta_error

    ####################################################################################################################

    def _calculate_landy_szalay(self, tc_random_catalog: treecorr.catalog.Catalog) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        dr = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        dr.process(self._tc_galaxy_catalog, tc_random_catalog)

        rr = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        rr.process(tc_random_catalog)

        ################################################################################################################

        wtheta, wtheta_variance = self._dd.calculateXi(rr = rr, dr = dr, rd = None)

        wtheta_error = np.sqrt(wtheta_variance)

        theta = np.exp(self._dd.meanlogr)

        ################################################################################################################

        return theta, wtheta, wtheta_error

########################################################################################################################
