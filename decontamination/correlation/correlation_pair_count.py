# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import healpy as hp

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
    Galaxy angular correlation function using the TreeCorr library. Supports pair counting (NN correlations)
    or a scalar field approach (KK correlations).

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
    bin_slop : typing.Optional[float]
        Precision parameter for binning (see `TreeCorr documentation <https://rmjarvis.github.io/TreeCorr/_build/html/binning.html#bin-slop>`_).
    footprint : typing.Optional[np.ndarray]
        HEALPix indices of the region where correlation must be calculated (KK correlations only).
    coverage : typing.Optional[np.ndarray]
        Observed sky fraction for each of the aforementioned HEALPix pixels (KK correlations only).
    nside : typing.Optional[np.ndarray]
        The HEALPix nside parameter (KK correlations only).
    nest : bool, default: True
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering (KK correlations only).
    """

    ####################################################################################################################

    def __init__(self, data_lon: np.ndarray, data_lat: np.ndarray, min_sep: float, max_sep: float, n_bins: int, bin_slop: typing.Optional[float] = None, footprint: typing.Optional[np.ndarray] = None, coverage: typing.Optional[np.ndarray] = None, nside: typing.Optional[bool] = None, nest: bool = True):

        ################################################################################################################

        if treecorr is None:

            raise ImportError('TreeCorr is not installed.')

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        self._bin_slop = bin_slop
        self._footprint = footprint
        self._coverage = coverage
        self._nside = nside
        self._nest = nest

        ################################################################################################################
        # BUILD THE CATALOG                                                                                            #
        ################################################################################################################

        self._data_catalog = self._build_catalog(data_lon, data_lat)

        ################################################################################################################
        # CORRELATE IT                                                                                                 #
        ################################################################################################################

        self._dd = self._correlate(self._data_catalog, None)

    ####################################################################################################################

    def _build_catalog(self, lon, lat):

        if self._footprint is None or self._nside is None:

            ############################################################################################################
            # NN CORRELATION                                                                                           #
            ############################################################################################################

            return treecorr.Catalog(
                ra = lon,
                dec = lat,
                ra_units = 'degrees',
                dec_units = 'degrees'
            )

            ############################################################################################################

        else:

            ############################################################################################################
            # ΚΚ CORRELATION                                                                                           #
            ############################################################################################################

            data_contrast = correlation_abstract.Correlation_Abstract._build_full_sky_contrast(self._nside, self._nest, self._footprint, lon, lat)

            lon, lat = hp.pix2ang(self._nside, self._footprint, nest = self._nest, lonlat = True)

            ############################################################################################################

            return treecorr.Catalog(
                ra = lon,
                dec = lat,
                ra_units = 'degrees',
                dec_units = 'degrees',
                k = data_contrast[self._footprint],
                w = self._coverage or np.ones(self._footprint.shape[0], dtype = np.float32)
            )

    ####################################################################################################################

    def _correlate(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> 'treecorr.NNCorrelation':

        ################################################################################################################

        if catalog1.k is None:

            if self._bin_slop is None:
                result = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
            else:
                result = treecorr.NNCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, bin_slop = self._bin_slop, sep_units = 'arcmin')

        else:

            if self._bin_slop is None:
                result = treecorr.KKCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
            else:
                result = treecorr.KKCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, bin_slop = self._bin_slop, sep_units = 'arcmin')

        ################################################################################################################

        result.process(catalog1, catalog2)

        ################################################################################################################

        return result

    ####################################################################################################################

    def calculate(self, estimator: str, random_lon: typing.Optional[np.ndarray] = None, random_lat: typing.Optional[np.ndarray] = None, random_kappa: typing.Optional[np.ndarray] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        if estimator == 'dd':

            return self._calculate_xy(self._data_catalog, None)

        ################################################################################################################

        if random_lon is None\
           or                \
           random_lat is None:

            raise ValueError(f'Parameters `random_lon` and `random_lat` have to be provided with estimator `{estimator}`.')

        ################################################################################################################

        self_random_catalog = self._build_catalog(random_lon, random_lat)

        ################################################################################################################

        if estimator == 'rr':
            return self._calculate_xy(self_random_catalog, None)
        if estimator == 'dr':
            return self._calculate_xy(self._data_catalog, self_random_catalog)
        if estimator == 'rd':
            return self._calculate_xy(self_random_catalog, self._data_catalog)

        else:

            if self._data_catalog.k is None:

                if estimator == 'peebles_hauser':
                    return self._calculate_xi(self_random_catalog, False, False)
                if estimator == 'landy_szalay_1':
                    return self._calculate_xi(self_random_catalog, True, False)
                if estimator == 'landy_szalay_2':
                    return self._calculate_xi(self_random_catalog, True, True)

            else:

                raise ValueError('Invalid estimator for kappa-kappa correlations (`peebles_hauser`, `landy_szalay1`, `landy_szalay_2` are forbidden)')

        ################################################################################################################

        raise ValueError('Invalid estimator (`dd`, `rr`, `dr`, `rd`, `peebles_hauser`, `landy_szalay1`, `landy_szalay_2` are authorized)')

    ####################################################################################################################

    def _calculate_xy(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        if catalog1 is not self._data_catalog or catalog2 is not None:

            xy = self._correlate(catalog1, catalog2)

        else:

            xy = self._dd

        ################################################################################################################

        theta = np.exp(xy.meanlogr)

        ################################################################################################################

        if catalog1.k is None:

            ############################################################################################################
            # NN CORRELATION                                                                                           #
            ############################################################################################################

            xi_theta = np.diff(xy.npairs, prepend = 0)

            xi_theta_error = np.zeros_like(xi_theta)

            ############################################################################################################

        else:

            ############################################################################################################
            # ΚΚ CORRELATION                                                                                           #
            ############################################################################################################

            xi_theta = xy.xi

            xi_theta_error = np.sqrt(xy.varxi)

        ################################################################################################################

        return theta, xi_theta, xi_theta_error

    ####################################################################################################################

    def _calculate_xi(self, self_random_catalog: 'treecorr.Catalog', with_dr: bool, with_rd: bool) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

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
