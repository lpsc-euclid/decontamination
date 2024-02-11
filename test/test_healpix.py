#!/usr/bin/env python3 -m pytest
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys
import time

os.environ['NUMBA_DISABLE_JIT'] = '1'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import numpy as np
import healpy as hp

import decontamination

########################################################################################################################

try:

    import healpix

except ModuleNotFoundError:

    healpix = None

########################################################################################################################

def test_npix():

    for nside in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:

        assert decontamination.nside2npix(nside) == hp.nside2npix(nside)

        assert decontamination.npix2nside(decontamination.nside2npix(nside)) == nside

########################################################################################################################

def test_pixarea():

    for nside in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:

        assert decontamination.nside2pixarea(nside) == hp.nside2pixarea(nside)

########################################################################################################################

def test_resol():

    for nside in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:

        assert decontamination.nside2resol(nside) == hp.nside2resol(nside)

########################################################################################################################

def test_xyf():

    for nside in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:

        pixels0 = np.arange(hp.nside2npix(nside))

        x, y, f = decontamination.nest2xyf(nside, pixels0)

        pixels1 = decontamination.xyf2nest(nside, x, y, f)

        assert np.allclose(pixels0, pixels1)

########################################################################################################################

def test_ang2pix():

    time_dec = 0
    time_ref = 0

    for nside in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:

        θ = np.random.uniform(0.0, 1.0 * np.pi, 1000)
        ϕ = np.random.uniform(0.0, 2.0 * np.pi, 1000)

        t1 = time.perf_counter()
        pix_dec = decontamination.ang2pix(nside, θ, ϕ, lonlat = False)
        t2 = time.perf_counter()
        pix_hp = hp.ang2pix(nside, θ, ϕ, nest = True, lonlat = False)
        t3 = time.perf_counter()

        time_dec = t2 - t1
        time_ref = t3 - t2

        assert np.allclose(pix_dec, pix_hp)

    print(f'\ndec: {time_dec:.4e}, ref: {time_ref:.4e}')

########################################################################################################################

def test_rand_ang():

    decontamination.randang(1, np.arange(1))

    if healpix:

        time_dec = 0
        time_ref = 0

        for nside in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:

            pixels = np.arange(hp.nside2npix(nside))

            t1 = time.perf_counter()
            θ_dec, ϕ_dec = decontamination.randang(nside, pixels, lonlat = True, compat = True, rng = np.random.default_rng(seed = 33))
            t2 = time.perf_counter()
            θ_ref, ϕ_ref = healpix.randang(nside, pixels, lonlat = True, nest = True, rng = np.random.default_rng(seed = 33))
            t3 = time.perf_counter()

            time_dec = t2 - t1
            time_ref = t3 - t2

            assert np.allclose(θ_dec, θ_ref)
            assert np.allclose(ϕ_dec, ϕ_ref)

        print(f'\ndec: {time_dec:.4e}, ref: {time_ref:.4e}')

########################################################################################################################
