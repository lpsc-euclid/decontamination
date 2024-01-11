# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np
import healpy as hp

from . import correlation_abstract

########################################################################################################################

try:

    import xpol

except ImportError:

    xpol = None

########################################################################################################################

# noinspection PyPep8Naming
class Correlation_PowerSpectrum(correlation_abstract.Correlation_Abstract):

    """
    Angular correlation function from power spectrum.

    .. math::
        \\xi(\\theta)=\\frac{1}{4\\pi}\\sum_{l=0}^{2\\times\\text{nside}}(2l+1)\\,C_l\\,P_l(\\cos\\theta)

    Parameters
    ----------
    data_lon : np.ndarray
        Galaxy catalog longitudes (in degrees).
    data_lat : np.ndarray
        Galaxy catalog latitudes (in degrees).
    footprint : np.ndarray
        HEALPix indices of the region where correlation must be calculated.
    nside : int
        The HEALPix nside parameter.
    nest : bool
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering.
    min_sep : float
        Minimum galaxy separation being considered (in arcmins).
    max_sep : float
        Maximum galaxy separation being considered (in arcmins).
    n_bins : int
        Number of angular bins.
    l_max : int, default: 3*nside-1
        Maximum :math:`l` of the power spectrum.
    library : str, default: **'xpol'**
        Library to be used for calculating the :math:`C_l` inside the footprint (**'xpol'**, **'anafast'**).
    """

    ####################################################################################################################

    def __init__(self, data_lon: np.ndarray, data_lat: np.ndarray, footprint: np.ndarray, nside: int, nest: bool, min_sep: float, max_sep: float, n_bins: int, l_max = None, library: str = 'xpol'):

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        self._nside = nside

        self._library = library

        ################################################################################################################

        self._l = None

        self._l_max = (3 * nside - 1) if l_max is None else l_max

        ################################################################################################################

        self._theta_radian = np.linspace(
            np.deg2rad(min_sep / 60.0),
            np.deg2rad(max_sep / 60.0),
            n_bins
        )

        self._theta_arcmin = np.linspace(
            min_sep,
            max_sep,
            n_bins
        )

        ################################################################################################################

        self._footprint = hp.nest2ring(nside, footprint) if nest else footprint

        # Anafast works only with the ring ordering!

        ################################################################################################################
        # BUILD THE FOOTPRINT                                                                                          #
        ################################################################################################################

        self._full_sky_footprint = np.zeros(hp.nside2npix(nside), dtype = np.float32)

        self._full_sky_footprint[self._footprint] = 1.0

        ################################################################################################################
        # BUILD THE CONTRAST                                                                                           #
        ################################################################################################################

        self._data_contrast = correlation_abstract.Correlation_Abstract._build_full_sky_contrast(
            self._nside,
            False,
            self._footprint,
            data_lon,
            data_lat
        )

        ################################################################################################################
        # CORRELATE IT                                                                                                 #
        ################################################################################################################

        self._dd = self._correlate(self._data_contrast)

    ####################################################################################################################

    @property
    def footprint(self):

        return self._footprint

    ####################################################################################################################

    @property
    def nside(self):

        return self._nside

    ####################################################################################################################

    @property
    def library(self):

        return self._library

    ####################################################################################################################

    @property
    def l(self):

        return self._l

    ####################################################################################################################

    @property
    def l_max(self):

        return self._l_max

    ####################################################################################################################

    @property
    def data_spectrum(self):

        return self._dd[0]

    ####################################################################################################################

    @property
    def data_contrast(self):

        return self._data_contrast

    ####################################################################################################################

    def cell2power_spectrum(self, cl: np.ndarray) -> np.ndarray:

        """???"""

        return self._l * (self._l + 1.0) * cl

    ####################################################################################################################

    def cell2correlation(self, cl: np.ndarray) -> np.ndarray:

        """???"""

        a = np.cos(self._theta_radian)

        b = (2.0 * self._l + 1.0) * cl

        return np.polynomial.legendre.legval(a, b) / (4.0 * np.pi)

    ####################################################################################################################

    def _correlate(self, contrast1: np.ndarray, contrast2: typing.Optional[np.ndarray] = None) -> typing.Tuple[np.ndarray, np.ndarray]:

        ################################################################################################################
        # LIBRARY = XPOL                                                                                               #
        ################################################################################################################

        if self._library == 'xpol':

            if xpol is None:

                raise ImportError('Xpol is not installed.')

            ####

            bins = xpol.Bins.fromdeltal(0, self._l_max, 1)

            xp = xpol.Xpol(self._full_sky_footprint, bins = bins, polar = False)

            cl, _ = xp.get_spectra(
                m1 = contrast1,
                m2 = contrast2,
                pixwin = True,
                remove_dipole = False
            )

        ################################################################################################################
        # LIBRARY = ANAFAST                                                                                            #
        ################################################################################################################

        else:

            ma1 = hp.ma(contrast1)
            ma1.mask = np.logical_not(self._full_sky_footprint)

            if contrast2 is not None:

                ma2 = hp.ma(contrast2)
                ma2.mask = np.logical_not(self._full_sky_footprint)

                cl = hp.anafast(map1 = ma1.filled(), map2 = ma2.filled(), lmax = self._l_max, pol = False)

            else:

                cl = hp.anafast(map1 = ma1.filled(), map2 = None, lmax = self._l_max, pol = False)

        ################################################################################################################

        if self._l is None:

            self._l = np.arange(cl.shape[0], dtype = np.int64)

        ################################################################################################################

        return (
            self.cell2power_spectrum(cl),
            self.cell2correlation(cl),
        )

    ####################################################################################################################

    def calculate(self, estimator: str, random_lon: typing.Optional[np.ndarray] = None, random_lat: typing.Optional[np.ndarray] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        if estimator == 'dd':

            return self._calculate_xy(self._data_contrast, None)

        ################################################################################################################

        if random_lon is None\
           or                \
           random_lat is None:

            raise ValueError(f'Parameters `random_lon` and `random_lat` have be provided with estimator `{estimator}`.')

        ################################################################################################################

        random_contrast = correlation_abstract.Correlation_Abstract._build_full_sky_contrast(
            self._nside,
            False,
            self._footprint,
            random_lon,
            random_lat
        )

        ################################################################################################################

        if estimator == 'rr':
            return self._calculate_xy(random_contrast, None)
        if estimator == 'dr':
            return self._calculate_xy(self._data_contrast, random_contrast)
        if estimator == 'rd':
            return self._calculate_xy(random_contrast, self._data_contrast)
        if estimator == 'peebles_hauser':
            return self._calculate_xi(random_contrast, False, False)
        if estimator == 'landy_szalay_1':
            return self._calculate_xi(random_contrast, True, False)
        if estimator == 'landy_szalay_2':
            return self._calculate_xi(random_contrast, True, True)

        ################################################################################################################

        raise ValueError('Invalid estimator (`dd`, `rr`, `dr`, `rd`, `peebles_hauser`, `landy_szalay1`, `landy_szalay_2`)')

    ####################################################################################################################

    def _calculate_xy(self, contrast1: np.ndarray, contrast2: typing.Optional[np.ndarray] = None) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if contrast1 is not self._data_contrast or contrast2 is not None:

            xy = self._correlate(contrast1, contrast2)

        else:

            xy = self._dd

        ################################################################################################################

        return self._theta_arcmin, xy[1], np.zeros_like(xy[1])

    ####################################################################################################################

    def _calculate_xi(self, random_contrast: np.ndarray, with_dr: bool, with_rd: bool) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        dd = self._calculate_xy(self._data_contrast, None)
        rr = self._calculate_xy(  random_contrast  , None)

        if with_dr:

            dr = self._calculate_xy(self._data_contrast, random_contrast)

            if with_rd:

                rd = self._calculate_xy(random_contrast, self._data_contrast)

                return dd[0], (dd[1] - dr[1] - rd[1] + rr[1]) / rr[1], dd[2]

            else:

                return dd[0], (dd[1] - 2.0 * dr[1] + rr[1]) / rr[1], dd[2]
        else:

            return dd[0], dd[1] / rr[1] - 1.0, dd[2]

########################################################################################################################
