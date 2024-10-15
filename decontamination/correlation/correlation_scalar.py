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
    Galaxy-galaxy 2pt correlation functions (2PCF) using the TreeCorr library. Supports pair counting (≡ ΚΚ
    correlations).

    Parameters
    ----------
    nside : int
        HEALPix nside parameter.
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
    data_w : np.ndarray
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

        self._lon, self._lat = hp.pix2ang(nside, footprint, nest = nest, lonlat = True)

        ################################################################################################################

        self._bin_slop = bin_slop
        self._n_threads = n_threads

        ################################################################################################################

        self._data_catalog = self._build_catalog(data_field, data_w)

        self._random_field = self._build_catalog(random_field, random_w) \
                                        if random_field is not None else self._data_catalog

    ####################################################################################################################

    @property
    def nside(self) -> typing.Optional[int]:

        """The HEALPix nside parameter."""

        return self._nside

    ####################################################################################################################

    @property
    def nest(self) -> typing.Optional[bool]:

        """If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering."""

        return self._nest

    ####################################################################################################################

    @property
    def footprint(self) -> typing.Optional[np.ndarray]:

        """HEALPix indices of the region where correlation must be calculated."""

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

    def _build_catalog(self, k: np.ndarray, w: np.ndarray) -> 'treecorr.Catalog':

        return treecorr.Catalog(
            ra = self._lon,
            dec = self._lat,
            k = k,
            w = w,
            ra_units = 'degrees',
            dec_units = 'degrees'
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

        ################################################################################################################

        raise ValueError('Invalid estimator (`dd`, `rr`, `dr`, `rd` are authorized)')

    ####################################################################################################################

    def _2pcf(self, catalog1: 'treecorr.Catalog', catalog2: typing.Optional['treecorr.Catalog'] = None) -> 'treecorr.KKCorrelation':

        kk = treecorr.KKCorrelation(bin_slop = self._bin_slop, min_sep = self._min_sep, max_sep = self._max_sep, nbins = self._n_bins, sep_units = 'arcmin')

        kk.process(catalog1, catalog2, num_threads = self._n_threads)

        return np.exp(kk.meanlogr), kk.xi, np.sqrt(kk.varxi)

########################################################################################################################
