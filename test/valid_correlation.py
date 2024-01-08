#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import tqdm

import numpy as np
import healpy as hp

import decontamination

from astropy.io import fits

import matplotlib.pyplot as plt

########################################################################################################################

if __name__ == '__main__':

    ####################################################################################################################

    nside = 128

    npix = hp.nside2npix(nside)

    np.random.seed(0)

    cl = np.random.uniform(size = 2 * nside)

    skymap = hp.synfast(cl, nside)

    min = np.min(skymap)
    max = np.max(skymap)

    skymap = (skymap - min) / (max - min)

    hp.mollview(skymap)
    plt.show()

    ####################################################################################################################

    footprint = np.arange(skymap.shape[0], dtype = np.int64)

    ####################################################################################################################

    generator = decontamination.Generator_NumberDensity(nside, footprint, nest = True, lonlat = True)

    catalog = np.empty(0, dtype = [('ra', np.float32), ('dec', np.float32)])

    for lon, lat in tqdm.tqdm(generator.generate(skymap, 2.0, n_max_per_batch = 1000)):

        rows = np.empty(lon.shape[0], dtype = catalog.dtype)
        rows['ra'] = lon
        rows['dec'] = lat

        catalog = np.append(catalog, rows)

    ####################################################################################################################

    correlation1 = decontamination.Correlation_PairCount(catalog['ra'], catalog['dec'], 3.0, 1000.0, 300)

    theta1, w_theta1, _ = correlation1.calculate('dd')

    ####################################################################################################################

    #correlation2 = decontamination.Correlation_PowerSpectrum(catalog['ra'], catalog['dec'], footprint, nside, False, 3.0, 1000.0, 300, library = 'anafast')
    correlation2 = decontamination.Correlation_PowerSpectrum(catalog['ra'], catalog['dec'], footprint, nside, False, 3.0, 1000.0, 300, library = 'xpol')

    theta2, w_theta2, _ = correlation2.calculate('dd')

    plt.scatter(x = theta1, y = theta1 * w_theta1)
    plt.xlabel('θ [arcmin]')
    plt.ylabel('θw(θ) (pair count)')
    #plt.semilogx()
    #plt.semilogy()
    plt.grid()
    plt.show()

    plt.scatter(x = theta2, y = theta2 * w_theta2)
    plt.xlabel('θ [arcmin]')
    plt.ylabel('θw(θ) (power spectrum)')
    #plt.semilogx()
    #plt.semilogy()
    plt.grid()
    plt.show()

    plt.plot(correlation2.ell, correlation2.spectrum)
    plt.xlabel('ell')
    plt.ylabel('power spectrum')
    plt.grid()
    plt.show()

########################################################################################################################
