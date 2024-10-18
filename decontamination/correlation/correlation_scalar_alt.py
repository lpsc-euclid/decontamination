# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import math
import tqdm
import typing

import numpy as np
import numba as nb
import healpy as hp

from . import correlation_abstract

from .. import jit, algo, device_array_zeros, get_max_gpu_threads

########################################################################################################################

# noinspection PyPep8Naming, PyTypeChecker, DuplicatedCode
class Correlation_ScalarAlt(correlation_abstract.Correlation_Abstract):

    """
    Galaxy-galaxy 2pt correlation functions (2PCF), standalone implementation. Scalar-scalar (≡ ΚΚ) correlations.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter.
    nest : bool
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering.
    footprint : np.ndarray
        HEALPix indices of the region where correlation must be calculated.
    data1 : np.ndarray
        ???
    min_sep : float
        Minimum galaxy separation being considered (in arcmins).
    max_sep : float
        Maximum galaxy separation being considered (in arcmins).
    n_bins : int
        Number of angular bins.
    data2 : np.ndarray, default: **None**
        ???
    data1_weights : np.ndarray, default: **None**
        ???
    data2_weights : np.ndarray, default: **None**
        ???
    """

    ####################################################################################################################

    def __init__(self, nside: int, nest: bool, footprint: np.ndarray, data1: np.ndarray, min_sep: float, max_sep: float, n_bins: int, data2: typing.Optional[np.ndarray] = None, data1_weights: typing.Optional[np.ndarray] = None, data2_weights: typing.Optional[np.ndarray] = None):

        ################################################################################################################

        super().__init__(min_sep, max_sep, n_bins)

        ################################################################################################################

        self._nest = nest
        self._nside = nside
        self._footprint = footprint

        lon, lat = hp.pix2ang(nside, footprint, nest = nest, lonlat = True)

        self._lon = np.radians(lon)
        self._sin_lat = np.sin(np.radians(lat))
        self._cos_lat = np.cos(np.radians(lat))

        ################################################################################################################

        self._log_min_sep = math.log(self._min_sep_rad)
        self._log_max_sep = math.log(self._max_sep_rad)

        self._inv_bin_width = n_bins / (
            self._log_max_sep
            -
            self._log_min_sep
        )

        ################################################################################################################

        self._bins = np.linspace(self._log_min_sep, self._log_max_sep, n_bins + 1)

        ################################################################################################################

        self._data1 = data1
        self._data1_weights = data1_weights
        self._data2 = data2
        self._data2_weights = data2_weights

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

    def calculate(self, estimator: str, enable_gpu = True) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        if estimator == 'dd':
            return self._2pcf(self._data1, self._data1, self._data1_weights, self._data1_weights, True, enable_gpu)
        if estimator == 'rr':
            return self._2pcf(self._data2, self._data2, self._data2_weights, self._data2_weights, True, enable_gpu)
        if estimator == 'dr':
            return self._2pcf(self._data1, self._data2, self._data1_weights, self._data2_weights, False, enable_gpu)
        if estimator == 'rd':
            return self._2pcf(self._data2, self._data1, self._data2_weights, self._data1_weights, False, enable_gpu)

        ################################################################################################################

        raise ValueError('Invalid estimator (`dd`, `rr`, `dr`, `rd` are authorized)')

    ####################################################################################################################

    def _2pcf(self, data1: np.ndarray, data2: np.ndarray, data1_weight: typing.Optional[np.ndarray], data2_weights: typing.Optional[np.ndarray], is_autocorr: bool, enable_gpu: bool) -> typing.Tuple[np.ndarray, np.ndarray]:

        ################################################################################################################

        if is_autocorr:
            if data1_weight is None or data2_weights is None:
                data1_weight = data2_weights = np.ones_like(data1)
        else:
            if data1_weight is None:
                data1_weight = np.ones_like(data1)
            if data2_weights is None:
                data2_weights = np.ones_like(data2)

        ################################################################################################################

        n_chunks = 100

        chunk_size = max(1, data1.shape[0] // n_chunks)

        threads_per_blocks = int(math.sqrt(get_max_gpu_threads()))

        ################################################################################################################

        device_w = device_array_zeros(shape = self.n_bins, dtype = np.float64)
        device_logr = device_array_zeros(shape = self.n_bins, dtype = np.float64)
        device_h = device_array_zeros(shape = self.n_bins, dtype = np.float64)

        for start, stop in tqdm.tqdm(algo.batch_iterator(data1.shape[0], chunk_size), total = n_chunks):

            compute_2pcf_step1[enable_gpu, (threads_per_blocks, threads_per_blocks), (stop - start, data2.shape[0])](
                device_w,
                device_logr,
                device_h,
                self._lon[start: stop],
                self._lon,
                self._sin_lat[start: stop],
                self._sin_lat,
                self._cos_lat[start: stop],
                self._cos_lat,
                data1[start: stop],
                data2,
                data1_weight[start: stop],
                data2_weights,
                self._min_sep_rad,
                self._max_sep_rad,
                self._log_min_sep,
                self._inv_bin_width,
                is_autocorr
            )

        result_w = device_w.copy_to_host()
        result_logr = device_logr.copy_to_host()
        result_h = device_h.copy_to_host()

        ################################################################################################################

        valid_bins = result_h > 0

        ################################################################################################################

        result_w   [valid_bins] /= result_h[valid_bins]
        result_logr[valid_bins] /= result_h[valid_bins]

        ################################################################################################################

        for idx in np.nonzero(~valid_bins)[0]:

            if idx + 1 < self._bins.shape[0]:

                result_logr[idx] = 0.5 * (
                    self._bins[idx + 0]
                    +
                    self._bins[idx + 1]
                )

        ################################################################################################################

        return np.degrees(np.exp(result_logr)) * 60.0, result_w, np.zeros_like(result_w)

########################################################################################################################

@jit(kernel = True, fastmath = True, parallel = True)
def compute_2pcf_step1(result_w: np.ndarray, result_logr: np.ndarray, result_h: np.ndarray, lon1: np.ndarray, lon2: np.ndarray, sin_lat1: np.ndarray, sin_lat2: np.ndarray, cos_lat1: np.ndarray, cos_lat2: np.ndarray, data1: np.ndarray, data2: np.ndarray, data1_weight: np.ndarray, data2_weight: np.ndarray, min_sep: float, max_sep: float, log_min_sep: float, inv_bin_width: float, is_autocorr: bool) -> None:

    if jit.is_gpu:

        ################################################################################################################
        # GPU                                                                                                          #
        ################################################################################################################

        i, j = jit.grid(2)

        if is_autocorr:

            if i < data1.shape[0] and i < j < data2.shape[0]:

                compute_2pcf_step2_xpu(result_w, result_logr, result_h, lon1[i], lon2[j], sin_lat1[i], sin_lat2[j], cos_lat1[i], cos_lat2[j], data1[i], data2[j], data1_weight[i], data2_weight[j], min_sep, max_sep, log_min_sep, inv_bin_width)

        else:

            if i < data1.shape[0] and j < data2.shape[0]:

                compute_2pcf_step2_xpu(result_w, result_logr, result_h, lon1[i], lon2[j], sin_lat1[i], sin_lat2[j], cos_lat1[i], cos_lat2[j], data1[i], data2[j], data1_weight[i], data2_weight[j], min_sep, max_sep, log_min_sep, inv_bin_width)

        ################################################################################################################

    else:

        ################################################################################################################
        # CPU                                                                                                          #
        ################################################################################################################

        if is_autocorr:

            for i in nb.prange(data1.shape[0]):
                for j in range(i + 1, data2.shape[0]):

                    compute_2pcf_step2_xpu(result_w, result_logr, result_h, lon1[i], lon2[j], sin_lat1[i], sin_lat2[j], cos_lat1[i], cos_lat2[j], data1[i], data2[j], data1_weight[i], data2_weight[j], min_sep, max_sep, log_min_sep, inv_bin_width)

        else:

            for i in nb.prange(data1.shape[0]):
                for j in range(0, data2.shape[0]):

                    compute_2pcf_step2_xpu(result_w, result_logr, result_h, lon1[i], lon2[j], sin_lat1[i], sin_lat2[j], cos_lat1[i], cos_lat2[j], data1[i], data2[j], data1_weight[i], data2_weight[j], min_sep, max_sep, log_min_sep, inv_bin_width)

        ################################################################################################################

    jit.syncthreads()

########################################################################################################################

@jit(kernel = False, inline = True, fastmath = True)
def _clip_xpu(x, a, b):

    return max(a, min(x, b))

########################################################################################################################

@jit(kernel = False, inline = True, fastmath = True)
def compute_2pcf_step2_xpu(result_w: np.ndarray, result_logr: np.ndarray, result_h: np.ndarray, lon1: np.ndarray, lon2: np.ndarray, sin_lat1: np.ndarray, sin_lat2: np.ndarray, cos_lat1: np.ndarray, cos_lat2: np.ndarray, data1: np.ndarray, data2: np.ndarray, data1_weight: np.ndarray, data2_weight: np.ndarray, min_sep: float, max_sep: float, log_min_sep: float, inv_bin_width: float) -> None:

    ####################################################################################################################

    cos_angle = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * math.cos(lon1 - lon2)

    sep = math.acos(_clip_xpu(cos_angle, -1.0, +1.0))

    if min_sep <= sep <= max_sep:

        ################################################################################################################

        log_sep = math.log(sep)

        bin_idx = int((log_sep - log_min_sep) * inv_bin_width)

        ################################################################################################################

        weight = data1_weight * data2_weight

        jit.atomic_add(result_w, bin_idx, weight * data1 * data2)

        jit.atomic_add(result_logr, bin_idx, weight * log_sep)

        jit.atomic_add(result_h, bin_idx, weight)

########################################################################################################################
