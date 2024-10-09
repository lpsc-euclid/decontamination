# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
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
class Correlation_Scalar(correlation_abstract.Correlation_Abstract):

    """
    Galaxy angular correlation function using the TreeCorr library. Supports pair counting (≡ NN correlations)
    or a scalar field approach (≡ KK correlations) if a footprint is provided.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (KK correlations only).
    nest : bool
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering.
    footprint : np.ndarray
        HEALPix indices of the region where correlation must be calculated.
    data_field : np.ndarray
        ???
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
    random_field : np.ndarray
        ???
    """

    ####################################################################################################################

    def __init__(self, nside: int, nest: bool, footprint: np.ndarray, data_field: np.ndarray, min_sep: float, max_sep: float, n_bins: int, bin_slop: typing.Optional[float] = None, n_threads: typing.Optional[int] = None, random_field: typing.Optional[np.ndarray] = None, data_w: typing.Optional[np.ndarray] = None, random_w: typing.Optional[np.ndarray] = None):

        ################################################################################################################

        if treecorr is None:

            raise ImportError('TreeCorr is not installed.')

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        self._nest = nest
        self._nside = nside
        self._footprint = footprint

        self._lon, self._lat = hp.pix2ang(self._nside, self._footprint, nest = self._nest, lonlat = True)

        self._bin_slop = bin_slop
        self._n_threads = n_threads

        ################################################################################################################
        # BUILD THE CATALOG                                                                                            #
        ################################################################################################################

        self._data_catalog = self._build_catalog(data_field, data_w)

        if random_field is not None:

            self._random_catalog = self._build_catalog(random_field, random_w)

        else:

            self._random_catalog = None

    ####################################################################################################################

    @property
    def nside(self) -> typing.Optional[int]:

        """The HEALPix nside parameter (KK correlations only)."""

        return self._nside

    ####################################################################################################################

    @property
    def nest(self) -> typing.Optional[bool]:

        """If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering (KK correlations only)."""

        return self._nest

    ####################################################################################################################

    @property
    def footprint(self) -> typing.Optional[np.ndarray]:

        """HEALPix indices of the region where correlation must be calculated (KK correlations only)."""

        return self._footprint

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

    def _build_catalog(self, field: np.ndarray, w: np.ndarray) -> 'treecorr.Catalog':

        return treecorr.Catalog(
            ra = self._lon,
            dec = self._lat,
            ra_units = 'degrees',
            dec_units = 'degrees',
            k = field,
            w = w,
        )

    ####################################################################################################################

    def _correlate(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> typing.Union['treecorr.NNCorrelation', 'treecorr.KKCorrelation']:

        ################################################################################################################

        if self._bin_slop is None:
            result = treecorr.KKCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')
        else:
            result = treecorr.KKCorrelation(min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, bin_slop = self._bin_slop, sep_units = 'arcmin')

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

        ################################################################################################################

        theta = np.exp(dd.meanlogr)

        ################################################################################################################

        if with_dr:

            dr = self._correlate(self._data_catalog, self._random_catalog)

            if with_rd:

                rd = self._correlate(self._random_catalog, self._data_catalog)

                ##

                xi_theta = (dd.npairs - dr.npairs - rd.npairs + rr.npairs) / rr.npairs

                xi_theta_error = np.zeros_like(xi_theta)

            else:

                xi_theta = (dd.npairs - 2 * dr.npairs + rr.npairs) / rr.npairs

                xi_theta_error = np.zeros_like(xi_theta)

        else:

            xi_theta = (dd.npairs / rr.npairs) - 1.0

            xi_theta_error = np.zeros_like(xi_theta)

        ################################################################################################################

        return theta, xi_theta, xi_theta_error

########################################################################################################################
