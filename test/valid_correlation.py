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

import matplotlib.pyplot as plt

########################################################################################################################

if __name__ == '__main__':

    nside = 128

    generator = decontamination.Generator_FullSkyUniform(nside, seed = 0)

    catalog = np.empty(0, dtype = [('ra', np.float32), ('dec', np.float32)])

    for lon, lat in tqdm.tqdm(generator.generate(2.5, n_max_per_batch = 1000)):

        rows = np.empty(lon.shape[0], dtype = catalog.dtype)
        rows['ra'] = lon
        rows['dec'] = lat

        catalog = np.append(catalog, rows)

    footprint = np.arange(hp.nside2npix(nside), dtype = np.int64)

    print('Number of galaxies:', catalog.shape[0])
    print('Number of pixels:', footprint.shape[0])

    if False:

        correlation = decontamination.Correlation_PairCount(catalog['ra'], catalog['dec'], 3.0, 250.0, 20)

        theta, w_theta, _ = correlation.calculate('dd')

    else:

        correlation = decontamination.Correlation_PowerSpectrum(catalog['ra'], catalog['dec'], footprint, nside, 3.0, 250.0, 20, library = 'xpol')

        theta, w_theta, _ = correlation.calculate('dd')

    plt.scatter(x = theta, y = theta * w_theta)
    plt.xlabel(r'$ \theta [arcmin] $')
    plt.ylabel(r'$ \theta \ w(\theta) $')
    plt.semilogx()
    #plt.semilogy()
    plt.grid()

    plt.show()

########################################################################################################################
