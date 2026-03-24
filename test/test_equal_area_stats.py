#!/usr/bin/env python3 -m pytest
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import pytest
import decontamination

import numpy as np

########################################################################################################################

def _make_generator(data, chunk_size):

    def builder():

        def generator():

            n = data.shape[1]

            for i in range(0, n, chunk_size):

                yield data[:, i:i + chunk_size]

        return generator

    return builder

########################################################################################################################

def _assert_edges_valid(edges, minima, maxima):

    assert np.all(np.isfinite(edges))

    assert np.all(edges[:, 0] == minima)
    assert np.all(edges[:, -1] == maxima)

    # monotonic
    assert np.all(np.diff(edges, axis = 1) >= 0.0)

########################################################################################################################

def _assert_centers_valid(centers, edges):

    for i in range(centers.shape[0]):

        for j in range(centers.shape[1]):

            if np.isfinite(centers[i, j]):

                assert edges[i, j] <= centers[i, j] <= edges[i, j + 1]

########################################################################################################################

@pytest.mark.parametrize('n_bins', [20, None])
def test_exact_vs_approx(n_bins):

    n = 5000

    rng = np.random.default_rng(12345)

    theta = np.linspace(0.0, 1.0 * np.pi, n, dtype = np.float64)
    phi = np.linspace(0.0, 6.0 * np.pi, n, dtype = np.float64)

    dipole, quadrupole = np.cos(theta), 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)

    noise1 = 0.05 * rng.standard_normal(n)
    noise2 = 0.05 * rng.standard_normal(n)
    noise3 = 0.05 * rng.standard_normal(n)

    systematic1 = 1.0 + 0.40 * dipole + noise1
    systematic2 = 2.0 + 0.30 * quadrupole + noise2
    systematic3 = 0.5 + 0.25 * np.sin(phi) + noise3

    data = np.vstack([
        systematic1,
        systematic2,
        systematic3,
    ])

    ####################################################################################################################

    exact = decontamination.Decontamination_Abstract.compute_equal_area_binning_and_statistics(
        data,
        n_bins = 20,
        exact = True
    )

    approx = decontamination.Decontamination_Abstract.compute_equal_area_binning_and_statistics(
        _make_generator(data, 512),
        n_bins = 20,
        exact = False
    )

    ####################################################################################################################

    edges_e, centers_e, minima_e, maxima_e, means_e, rmss_e, stds_e, n_vectors_e = exact
    edges_a, centers_a, minima_a, maxima_a, means_a, rmss_a, stds_a, n_vectors_a = approx

    ####################################################################################################################

    assert np.allclose(edges_e, edges_a, atol = 1e-3)
    assert np.allclose(centers_e, centers_a, atol = 1e-3)

    assert np.allclose(minima_e, minima_a, atol = 1.0e-12)
    assert np.allclose(maxima_e, maxima_a, atol = 1.0e-12)

    assert np.allclose(means_e, means_a, atol = 1.0e-12)
    assert np.allclose(rmss_e, rmss_a, atol = 1.0e-12)
    assert np.allclose(stds_e, stds_a, atol = 1.0e-12)

########################################################################################################################

def test_nan_rejection():

    dim = 2

    n_vectors = 1000

    data = np.random.randn(dim, n_vectors)

    ####################################################################################################################

    data[0, 100: 200] = np.nan

    ####################################################################################################################

    edges, centers, minima, maxima, means, rmss, stds, n = decontamination.Decontamination_Abstract.compute_equal_area_binning_and_statistics(data, n_bins = 10)

    ####################################################################################################################

    assert n_vectors == n + 100

########################################################################################################################

def test_generator_vs_array():

    dim = 3

    n_vectors = 3000

    data = np.random.randn(dim, n_vectors)

    ####################################################################################################################

    res_array = decontamination.Decontamination_Abstract.compute_equal_area_binning_and_statistics(
        data,
        n_bins = 15,
        exact = True
    )

    res_gen = decontamination.Decontamination_Abstract.compute_equal_area_binning_and_statistics(
        _make_generator(data, 256),
        n_bins = 15,
        exact = False
    )

    ####################################################################################################################

    for i in range(2, 7):

        assert np.allclose(res_array[i], res_gen[i], atol = 1e-2)

########################################################################################################################

def test_edges_and_centers_validity():

    dim = 2

    n_vectors = 2000

    data = np.random.randn(dim, n_vectors)

    ####################################################################################################################

    edges, centers, minima, maxima, *_ = decontamination.Decontamination_Abstract.compute_equal_area_binning_and_statistics(data, n_bins = 25)

    ####################################################################################################################

    _assert_edges_valid(edges, minima, maxima)

    _assert_centers_valid(centers, edges)

########################################################################################################################
