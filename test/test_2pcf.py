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

    ####################################################################################################################

    mean = np.sum(kappa1) / np.sum(w1)

    contrast = (kappa1 / w1 - mean) / mean

    correlator = decontamination.Correlation_Scalar(nside = nside, nest = True, footprint = footprint, data_field = contrast, bin_slop = 0, min_sep = min_sep, max_sep = max_sep, n_bins = n_bins, data_w = w1)

    theta_mean, w, _ = correlator.calculate(estimator = 'dd')

    ####################################################################################################################

    print(theta_mean)
    print(w)

    assert np.allclose(theta_mean, expected_theta, rtol = 1e-4)
    assert np.allclose(w, expected_w, rtol = 1e-4)

########################################################################################################################

def test_2pcf_pair_count():

    expected_theta = np.array([1.29058237, 2.0099951, 3.13488704, 4.86697839, 7.58812258, 11.81448774, 18.34868782, 28.37480589, 43.33409523, 68.98899415, 109.30914378, 170.31072838, 265.59909124, 413.83079737])

    expected_w = np.array([2.07549282, 2.07830542, 2.13897731, 2.08079126, 2.02110906, 1.92698776, 1.81046642, 1.5346265, 0.89517858, 0.0682243, -0.01652028, -0.01174492, 0.00760077, -0.00595221])

    ####################################################################################################################
    # MEAN                                                                                                             #
    ####################################################################################################################

    mean_density = np.mean(kappa1)

    ####################################################################################################################
    # DATA                                                                                                             #
    ####################################################################################################################

    data_catalog = np.empty(0, dtype=[('ra', np.float32), ('dec', np.float32)])

    generator = decontamination.Generator_NumberDensity(nside, footprint, nest = True, seed = 500)

    for lon, lat in generator.generate(kappa1, mult_factor = 10.0 / mean_density, n_max_per_batch = 10_000):

        rows = np.empty(lon.shape[0], dtype=data_catalog.dtype)
        rows['ra'] = lon
        rows['dec'] = lat

        data_catalog = np.append(data_catalog, rows)

    ####################################################################################################################
    # UNIFORM                                                                                                          #
    ####################################################################################################################

    uniform_catalog = np.empty(0, dtype=[('ra', np.float32), ('dec', np.float32)])

    generator = decontamination.Generator_NumberDensity(nside, footprint, nest = True, seed = 5000)

    for lon, lat in generator.generate(w1, mult_factor = 10.0, n_max_per_batch = 10_000):

        rows = np.empty(lon.shape[0], dtype = uniform_catalog.dtype)
        rows['ra'] = lon
        rows['dec'] = lat

        uniform_catalog = np.append(uniform_catalog, rows)

    ####################################################################################################################
    # 2PCF                                                                                                             #
    ####################################################################################################################

    correlator = decontamination.Correlation_PairCount(data_catalog['ra'], data_catalog['dec'], random_lon = uniform_catalog['ra'], random_lat = uniform_catalog['dec'], min_sep = min_sep, max_sep = max_sep, n_bins = n_bins, bin_slop = 0.01)

    theta_mean, w, _ = correlator.calculate('landy_szalay_1')

    ####################################################################################################################

    print(theta_mean)
    print(w)

    assert np.allclose(theta_mean, expected_theta, rtol = 1e-4)
    assert np.allclose(w, expected_w, rtol = 1e-4)

########################################################################################################################
