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

    uniform_ra = np.array([], dtype = np.float32)
    uniform_dec = np.array([], dtype = np.float32)

    generator2 = decontamination.Generator_FullSkyUniform(nside, seed = None)

    for lon, lat in tqdm.tqdm(generator2.generate(10.0, n_max_per_batch = 1000)):

        uniform_ra = np.concatenate((uniform_ra, lon))
        uniform_dec = np.concatenate((uniform_dec, lat))

    ####################################################################################################################

    n_catalog = len(catalog_ra)
    n_uniform = len(uniform_ra)

    print('#catalog', n_catalog)
    print('#uniform', n_uniform)

    ####################################################################################################################

    correlation1 = decontamination.Correlation_PairCount(catalog_ra, catalog_dec, 3.0, 1500.0, 300)

    theta1, w_theta1, _ = correlation1.calculate('dd')

    theta2, w_theta2, _ = correlation1.calculate(
        'peebles_hauser',
        random_lon = uniform_ra,
        random_lat = uniform_dec
    )

    ####################################################################################################################

    correlation2 = decontamination.Correlation_PairCount(catalog_ra, catalog_dec, 3.0, 1500.0, 300, footprint = footprint, nside = nside, nest = False)

    theta3, w_theta3, _ = correlation2.calculate('dd')

    ####################################################################################################################

    correlation3 = decontamination.Correlation_PowerSpectrum(catalog_ra, catalog_dec, footprint, nside, False, 3.0, 1500.0, 300, library ='anafast')
    #correlation3 = decontamination.Correlation_PowerSpectrum(catalog_ra, catalog_dec, footprint, nside, False, 3.0, 1500.0, 300, library = 'xpol')

    theta4, w_theta4, _ = correlation3.calculate('dd')

    ####################################################################################################################

    hp.mollview(correlation3.data_contrast)
    plt.show()

    ####################################################################################################################

    plt.scatter(x = theta4, y = theta4 * correlation3.cell2correlation(cl_th), label ='th')
    plt.scatter(x = theta4, y = theta4 * correlation3.cell2correlation(cl_reco), label ='reco')
    plt.xlabel('θ [arcmin]')
    plt.ylabel('θw(θ)')
    #plt.semilogx()
    #plt.semilogy()
    plt.grid()
    plt.legend()
    plt.show()

    plt.scatter(x = theta2, y = theta2 * w_theta2, label = 'NN')
    plt.scatter(x = theta3, y = theta3 * w_theta3, label = 'KK')
    plt.scatter(x = theta4, y = theta4 * w_theta4, label ='xpol')
    plt.xlabel('θ [arcmin]')
    plt.ylabel('θw(θ)')
    #plt.semilogx()
    #plt.semilogy()
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(correlation3.l, correlation3.cell2power_spectrum(cl_th), label = 'th')
    plt.plot(correlation3.l, correlation3.cell2power_spectrum(cl_reco), label = 'reco')
    plt.xlabel('l')
    plt.ylabel('power spectrum')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(correlation3.l, correlation3.data_spectrum, label = 'xpol')
    plt.xlabel('l')
    plt.ylabel('power spectrum')
    plt.grid()
    plt.legend()
    plt.show()

########################################################################################################################
