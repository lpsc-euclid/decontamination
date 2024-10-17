#~!/usr/bin/env python3 -m pytest
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys
import typing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import decontamination

import numpy as np
import healpy as hp

########################################################################################################################

nside = 64

########################################################################################################################

def generate_data(nside: int, fraction: float) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    ####################################################################################################################

    np.random.seed(42)

    ####################################################################################################################

    npix = hp.nside2npix(nside)

    footprint = np.arange(npix, dtype = np.int64)

    ####################################################################################################################

    lmax = 3 * nside - 1

    cls = np.ones(lmax + 1)

    kappa1 = hp.synfast(cls, nside = nside, lmax = lmax)
    kappa2 = hp.synfast(cls, nside = nside, lmax = lmax)

    ####################################################################################################################

    indices = np.random.choice(footprint, size = int(fraction * npix), replace = False)

    footprint_subset = footprint[indices]
    kappa1_subset = kappa1[indices]
    kappa2_subset = kappa2[indices]

    ####################################################################################################################

    w1_subset = np.random.uniform(0.9, 1.0, size = len(kappa1_subset))
    w2_subset = np.random.uniform(0.9, 1.0, size = len(kappa2_subset))

    ####################################################################################################################

    return footprint_subset, np.clip(kappa1_subset, 0, 1000), np.clip(kappa2_subset, 0, 1000), w1_subset, w2_subset

########################################################################################################################

min_sep = 1.0
max_sep = 500.0
n_bins = 14

footprint, kappa1, kappa2, w1, w2 = generate_data(nside, 0.25)

########################################################################################################################

def test_2pcf_scalar():

    expected_theta = np.array([1.24850931, 1.94614573, 3.03360429, 4.72870805, 7.37099426, 11.48972527, 17.90990228, 27.91751692, 48.67165203, 64.88142103, 113.89551151, 170.98324272, 262.89435072, 412.93908318])

    expected_w = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.06166733, -0.02861535, -0.01513083, -0.02068876, 0.01380232, -0.00733413])

    ################################################################################################################

    mean = np.sum(kappa1) / np.sum(w1)

    contrast = (kappa1 / w1 - mean) / mean

    correlator = decontamination.Correlation_Scalar(nside = nside, nest = True, footprint = footprint, data_field = contrast, bin_slop = 0, min_sep = min_sep, max_sep = max_sep, n_bins = n_bins, data_w = w1)

    theta_mean, w, _ = correlator.calculate(estimator = 'dd')

    ################################################################################################################

    print()
    print(theta_mean)
    print(w)

    assert np.allclose(theta_mean, expected_theta, rtol = 1e-4)
    assert np.allclose(w, expected_w, rtol = 1e-4)

########################################################################################################################

def test_2pcf_scalar_alt():

    if decontamination.CPU_OPTIMIZATION_AVAILABLE:

        expected_theta = np.array([1.24850931, 1.94614573, 3.03360429, 4.72870805, 7.37099426, 11.48972527, 17.90990228, 27.91751692, 48.67434388, 64.88393405, 113.89411226, 171.05373384, 262.87680752, 412.67380433])

        expected_w = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.07191689, -0.02853841, -0.01464235, -0.0227004, 0.0141867, -0.00724601])

        ################################################################################################################

        mean = np.sum(kappa1) / np.sum(w1)

        contrast = (kappa1 / w1 - mean) / mean

        correlator = decontamination.Correlation_ScalarAlt(nside = nside, nest = True, footprint = footprint, data_field = contrast, bin_slop = 0, min_sep = min_sep, max_sep = max_sep, n_bins = n_bins, data_w = w1)

        theta_mean, w, _ = correlator.calculate(estimator = 'dd')

        ################################################################################################################

        print()
        print(theta_mean)
        print(w)

        assert np.allclose(theta_mean, expected_theta, rtol = 1e-4)
        assert np.allclose(w, expected_w, rtol = 1e-4)

########################################################################################################################
