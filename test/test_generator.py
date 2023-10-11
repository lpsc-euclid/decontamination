#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import pytest
import decontamination

import numpy as np

import healpy as hp

########################################################################################################################

def test_generator():

    expected_lon = np.array([
        99.14578, 17.327156, -21.811174, -40.17346, 10.4683485,
        11.288566, 5.6166277, 7.702591, 114.74092, 108.70205,
        94.01152, 188.61621, 191.59125, 192.01852, 215.87967,
        266.8664, 280.44647, 277.98703, 308.47684, 267.31274,
        276.4862, 295.99734, 289.9911, 273.98764, 269.03305,
        301.08615, 47.806644, 44.72147, 62.038666, 8.755553,
        44.95626, 65.036804, 80.385414, 4.546078, 61.79718,
        27.64675, 39.921806, 160.34837, 142.7173, 141.83322,
        103.59922, 155.68407, 161.81384, 128.85085, 144.60216,
        242.95297, 215.10767, 218.73886, 181.45612, 197.97952,
        230.47252, 213.1391, 258.9438, 347.77795, 321.05753,
        311.88742, 356.0623, 355.3369, 327.94318, 325.24625,
        386.40714, 281.3371, 307.64127
    ], dtype = np.float32)

    expected_lat = np.array([
        32.30332, 23.456558, 7.086174, 0.62548065, 2.0698318,
        27.733608, -9.875534, 33.99485, 21.511406, 22.817726,
        -6.8113937, -14.716644, 10.1100235, 7.7379303, 6.2180786,
        31.59063, -10.626137, -30.97551, -7.0625916, 42.294113,
        -26.259941, -19.11621, -22.392265, 35.39735, -43.614227,
        -6.7157364, -52.095535, 1.5861206, -50.604034, -29.13205,
        -69.94238, -22.874878, -79.745636, -38.56259, -71.87674,
        -16.38179, -30.75235, -56.084473, -39.93358, -14.601448,
        -56.644867, -64.39354, -27.17923, -89.184586, -19.329803,
        -15.984901, -79.73274, -1.9304504, -58.70929, -26.291458,
        -54.218704, -30.494987, -27.64006, -36.111267, -17.088646,
        -21.833847, -36.59134, -39.32985, -29.391296, -16.515717,
        -82.06714, -51.579926, -17.805923
    ], dtype = np.float32)

    nside = 1

    npix = hp.nside2npix(nside)

    pixels = np.arange(npix, dtype = np.int32)
    weight = np.arange(npix, dtype = np.float32)

    generator = decontamination.Generator_FromDensity(nside, pixels, nest = True)

    lon, lat = generator.generate(weight, 1, seed = 0)

    assert np.allclose(lon, expected_lon)

    assert np.allclose(lat, expected_lat)

########################################################################################################################

if __name__ == '__main__':

    pytest.main([__file__])

########################################################################################################################