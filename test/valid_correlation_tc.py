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

    ####################################################################################################################

    nside = 64

    ####################################################################################################################

    l = np.arange(2 * nside + 1, dtype = np.float32)

    cl_th = np.exp(-(l - 50.0) ** 2 / (2.0 * 10.0 ** 2)) + np.cos(l / np.pi)

    skymap = hp.synfast(cl_th, nside)

    ####################################################################################################################

    cl_reco = hp.anafast(map1 = skymap, lmax = 2 * nside, pol = False)

    plt.plot(l, cl_th, label = 'th')
    plt.plot(l, cl_reco, label = 'reco')
    plt.xlabel('l')
    plt.ylabel('cl')
    plt.grid()
    plt.legend()
    plt.show()

    ####################################################################################################################

    min = np.min(skymap)
    max = np.max(skymap)

    skymap = (skymap - min) / (max - min)

    hp.mollview(skymap)
    plt.show()

    ####################################################################################################################

    footprint = np.arange(skymap.shape[0], dtype = np.int64)

    ####################################################################################################################

    catalog_ra = np.array([], dtype = np.float32)
    catalog_dec = np.array([], dtype = np.float32)

    generator1 = decontamination.Generator_FullSkyUniform(nside, seed = None)

    for lon, lat in tqdm.tqdm(generator1.generate(10.0, n_max_per_batch = 1000)):

        pix = hp.ang2pix(nside, lon, lat, nest = False, lonlat = True)

        n = np.random.uniform(size = pix.shape[0])

        ok = np.where((1.0 - skymap[pix]) < n)[0]

        catalog_ra = np.concatenate((catalog_ra, lon[ok]))
        catalog_dec = np.concatenate((catalog_dec, lat[ok]))

    ####################################################################################################################

    correlation1 = decontamination.Correlation_PairCount(catalog_ra, catalog_dec, 3.0, 1500.0, 300)

    theta1, w_theta1, _ = correlation1.calculate('dd')

    ####################################################################################################################

    plt.scatter(x = theta1, y = theta1 * w_theta1, label = 'treecorr')
    plt.xlabel('θ [arcmin]')
    plt.ylabel('θw(θ)')
    #plt.semilogx()
    #plt.semilogy()
    plt.grid()
    plt.legend()
    plt.show()

    ####################################################################################################################

    print('bye')

########################################################################################################################
