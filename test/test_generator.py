#!/usr/bin/env python3 -m pytest
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import decontamination

import numpy as np

import healpy as hp

########################################################################################################################

@pytest.mark.parametrize('n_max_per_batch', [None, 10])
def test_number_density_generator(n_max_per_batch):

    expected_lon = np.array([
        143.03794086, 9.27174579, -22.03107032, -16.79844353, -33.41689499,
        11.68266999, -13.41208807, -23.13960025, 93.3600235, 73.13026261,
        112.0004341, 191.71377674, 168.45869068, 165.60775249, 170.05874361,
        242.36628823, 268.17739824, 231.79797966, 271.07931872, 232.92608393,
        273.04027777, 281.14267519, 264.52576885, 257.72381391, 248.02809808,
        238.8827032, 86.02811654, 88.04239236, 56.03365256, 84.02381021,
        83.3898099, 40.00458252, 63.17236074, 46.54532921, 74.80120015,
        27.15020153, 46.44146532, 146.21556708, 127.01306068, 143.62265288,
        161.53529542, 133.1409289, 166.82523355, 112.34481188, 94.47694628,
        206.35110234, 182.38909002, 189.67761432, 205.23951804, 261.76429461,
        264.20955653, 222.27940947, 203.05742512, 336.00752214, 288.30224616,
        331.67703002, 285.72202723, 327.02312496, 318.04932254, 301.92640982,
        322.98597333, 342.60996012, 285.66875189
    ], dtype = np.float32)

    expected_lat = np.array([
        10.86189489, -15.34457699, -12.66653481, -1.11959217, 5.41788274,
        -20.6666138, 21.76754282, 15.26666174, -2.86122484, 24.21781615,
        15.82256678, 9.79369304, 12.12617554, 5.70613414, 25.39153764,
        -5.97156202, 36.64820328, -4.65193471, 38.72649498, 4.64194084,
        28.35107633, -11.05548081, 28.9212575, -7.46710059, 14.42899984,
        -8.72287134, -59.57532856, -55.06691409, -29.09069798, -42.05686229,
        -46.4833457, -53.12950856, -18.32886449, -66.41771489, -36.2049111,
        -70.59181514, -34.97304188, -24.82672846, -56.72545843, -21.73093619,
        -26.59484458, -34.67886332, -42.12965375, -27.10915438, -48.60299237,
        -30.2124665, -72.38752691, -55.26141065, -43.01176253, -38.28602854,
        -45.66439909, -51.35276105, -20.88287457, -37.61126796, -35.31200915,
        -23.91144568, -26.13735836, -66.50776261, -11.83209353, -14.42611346,
        -12.5538997, -36.49608799, -51.31489195
    ], dtype = np.float32)

    nside = 1

    npix = hp.nside2npix(nside)

    pixels = np.arange(npix, dtype = np.int32)
    weight = np.arange(npix, dtype = np.float32)

    generator = decontamination.Generator_NumberDensity(nside, pixels, nest = True, seed = 0)

    lon = np.empty(0, dtype = np.float32)
    lat = np.empty(0, dtype = np.float32)

    for _lon, _lat in generator.generate(weight, 1, n_max_per_batch = n_max_per_batch):

        lon = np.append(lon, _lon)
        lat = np.append(lat, _lat)

    assert np.allclose(lon, expected_lon)
    assert np.allclose(lat, expected_lat)

########################################################################################################################

@pytest.mark.parametrize('n_max_per_batch', [None, 10])
def test_uniform_generator(n_max_per_batch):

    expected_lon = np.array([
        15.92198526, 81.61761075, 236.23471978, 236.53749621, 193.75186869,
        207.1786702, 191.31170382, 250.31790695, 261.80622882, 323.36269171,
        -15.56932066, 98.03794086, 99.27174579, 67.96892968, 27.69434522,
        11.58310501, 69.82476572, 121.58791193, 111.86039975, 138.63196923,
        118.13026261, 157.0004341, 326.71377674, 303.45869068
    ], dtype = np.float32)

    expected_lat = np.array([
        60.45007571, 56.39116146, 36.55537956, 38.42386631, 55.90483215,
        34.53409102, 37.6695407, 43.16316526, 38.56251597, 18.92101192,
        9.46484532, -28.56949407, -15.34457699, -12.66653481, -43.3079757,
        -34.90710904, -67.84648637, -17.20700639, -23.78807778, -45.61984105,
        -14.86011593, -23.20408608, -29.77304346, -27.16802023
    ], dtype = np.float32)

    nside = 1

    npix = hp.nside2npix(nside)

    pixels = np.arange(npix, dtype = np.int32)

    generator = decontamination.Generator_Uniform(nside, pixels, nest = True, seed = 0)

    lon = np.empty(0, dtype = np.float32)
    lat = np.empty(0, dtype = np.float32)

    for _lon, _lat in generator.generate(2, n_max_per_batch = n_max_per_batch):

        lon = np.append(lon, _lon)
        lat = np.append(lat, _lat)

    assert np.allclose(lon, expected_lon)
    assert np.allclose(lat, expected_lat)

########################################################################################################################

@pytest.mark.parametrize('n_max_per_batch', [None])
def test_full_sky_generator(n_max_per_batch):

    expected_lon = np.array([
        -165.24953138, -174.05005121, 112.77728611, 148.59200782, 38.38887928,
        82.61876195, 15.70499693, 156.62607256, 113.70727948, -179.01413994,
        128.66553957, -167.90919289, 82.67596071, -116.76397658, 130.74441205,
        14.92603929, -72.10371941, -27.83260037, -169.80491839, -135.25802046,
        61.42478929, 52.98822417, 41.53864013, -41.87608047, 178.99557688,
        173.10072196
    ], dtype = np.float32)

    expected_lat = np.array([
        21.78248399, 17.51278186, 22.14138874, -12.83576458, -46.87021578,
        26.29395985, 2.90663792, -22.30372122, -1.6233655, 51.16688404,
        60.23709554, -16.52356117, 8.22493458, -20.87076132, 10.87109659,
        -18.9157545, -12.51892047, 51.31084171, -33.07147378, 14.26306451,
        -56.30156869, 41.70444642, 35.04339005, -31.41688261, 48.84817365,
        -61.98979128
    ], dtype = np.float32)

    nside = 1

    generator = decontamination.Generator_FullSkyUniform(nside, seed = 0)

    lon = np.empty(0, dtype = np.float32)
    lat = np.empty(0, dtype = np.float32)

    for _lon, _lat in generator.generate(2, n_max_per_batch):

        lon = np.append(lon, _lon)
        lat = np.append(lat, _lat)

    assert np.allclose(lon, expected_lon)
    assert np.allclose(lat, expected_lat)

########################################################################################################################
