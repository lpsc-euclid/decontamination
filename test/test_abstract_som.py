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

########################################################################################################################

som = decontamination.SOM_PCA(4, 4, 4, dtype = np.float32, topology = 'square')

som.init_rand(seed = 0)

########################################################################################################################

def test_distance_map():

    expected = np.array([
        [0.26441127, 0.61138950, 0.52692540, 0.22971392],
        [0.52755773, 0.72621530, 0.66814790, 0.47758296],
        [0.48859516, 0.66861326, 1.00000000, 0.47632787],
        [0.22816606, 0.53763765, 0.40897673, 0.29011133],
    ], dtype = np.float32)

    assert np.allclose(som.distance_map(), expected)

########################################################################################################################

@pytest.mark.parametrize('enable_gpu', [False, True])
@pytest.mark.capture(no = True)
def test_activation_map(enable_gpu):

    expected = np.array([
        [3529015, 543220, 886912, 1282646],
        [4292427, 574837, 1051709, 1129844],
        [1604622, 3419371, 1270512, 929928],
        [1917657, 400646, 614078, 1552576],
    ], dtype = np.int64)

    ##

    data = np.random.default_rng(seed = 0).random((25_000_000, 4), np.float32)

    ##

    assert np.allclose(som.activation_map(data, enable_gpu = enable_gpu, show_progress_bar = True), expected)

########################################################################################################################

@pytest.mark.parametrize('enable_gpu', [False, True])
def test_winners(enable_gpu):

    expected = np.array([
        0, 1, 2, 3,
        4, 5, 6, 7,
    ], dtype = np.int64)

    ##

    data = np.random.default_rng(seed = 0).random((8, 4), np.float32)

    ##

    assert np.allclose(som.winners(data, enable_gpu = enable_gpu), expected)

########################################################################################################################

if __name__ == '__main__':

    pytest.main([__file__])

########################################################################################################################
