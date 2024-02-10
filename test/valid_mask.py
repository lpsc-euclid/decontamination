#!/usr/bin/env python3 -m pytest
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import numpy as np

import decontamination

import matplotlib.pyplot as plt

########################################################################################################################

try:

    from astropy.io import fits as astropy_fits

    from astropy.wcs import WCS as astropy_WCS

except ModuleNotFoundError:

    astropy_fits = None

    astropy_WCS = None

########################################################################################################################

def test_pix2world():

    if astropy_fits and astropy_WCS:

        nside = 16384

        orders, indices = decontamination.nuniq_to_order_index(np.array([3181747, 3181748, 3181750, 3181752, 3181753, 3181755, 3181756, 3181757, 3181758, 3182097, 12726907, 12726983, 12726987, 12726996, 12726998, 12727004, 12727006, 12727017, 12727019, 12727036, 12728385, 12728387, 12728393, 12728396, 12728400, 50907610, 50907621, 50907622, 50907623, 50907625, 50907626, 50907627, 50907632, 50907634, 50907640, 50907642, 50907925, 50907926, 50907927, 50907929, 50907930, 50907931, 50907941, 50907942, 50907943, 50907945, 50907946, 50907947, 50908020, 50908022, 50908028, 50908030, 50908064, 50908065, 50908067, 50908073, 50908075, 50908148, 50908152, 50913537, 50913581, 50913583, 50913588, 50913592, 50913604, 50913608, 203630335, 203630399, 203630479, 203630483, 203630499, 203630540, 203630542, 203630564, 203630566, 203630572, 203630574, 203630575, 203631103, 203631445, 203631447, 203631453, 203631455, 203631477, 203631679, 203631695, 203631699, 203631715, 203631759, 203631763, 203631779, 203631962, 203631984, 203631986, 203631992, 203631994, 203632116, 203632118, 203632124, 203632126, 203632265, 203632267, 203632289, 203632291, 203632596, 203632597, 203632598, 203632600, 203632601, 203632602, 203632612, 203632613, 203632614, 203632616, 203632617, 203632618, 203635234, 203635240, 203635242, 203654156, 203654157, 203654159, 203654181, 203654183, 203654189, 203654191, 203654320, 203654321, 203654323, 203654329, 203654331, 203654356, 203654357, 203654358, 203654360, 203654361, 203654362, 203654372, 203654373, 203654374, 203654376, 203654377, 203654378, 203654420, 203654421, 203654422, 203654424, 203654425, 203654426, 203654436, 203654437, 203654438, 203654440, 203654441, 203654442, 203654673, 203654675, 203654676, 203654677, 203654678, 814521333, 814521334, 814521335, 814521337, 814521338, 814521339, 814521589, 814521590, 814521591, 814521593, 814521594, 814521595, 814521734, 814521737, 814521738, 814521739, 814521740, 814521742, 814521909, 814521910, 814521911, 814521913, 814521914, 814521915, 814521925, 814521926, 814521927, 814521929, 814521930, 814521931, 814521989, 814521990, 814521991, 814521993, 814521994, 814521995, 814522128, 814522130, 814522136, 814522138, 814522262, 814522268, 814522270, 814522292, 814522294, 814522295, 814524405, 814524406, 814524407, 814524409, 814524410, 814524411, 814525777, 814525779, 814525785, 814525787, 814525809, 814525917, 814525919, 814525941, 814525943, 814525949, 814526709, 814526710, 814526711, 814526713, 814526714, 814526715, 814526773, 814526774, 814526775, 814526777, 814526778, 814526779, 814526789, 814526790, 814526791, 814526793, 814526794, 814526795, 814526853, 814526854, 814526855, 814526857, 814526858, 814526859, 814527029, 814527030, 814527031, 814527033, 814527034, 814527035, 814527045, 814527046, 814527047, 814527049, 814527050, 814527051, 814527109, 814527110, 814527111, 814527113, 814527114, 814527115, 814527810, 814527816, 814527818, 814527840, 814527842, 814527948, 814527950, 814527972, 814527974, 814527980, 814527982, 814527983, 814528346, 814528368, 814528370, 814528376, 814528378, 814528379, 814528500, 814528502, 814528508, 814528510, 814529057, 814529059, 814529065, 814529067, 814529188, 814529189, 814529191, 814529197, 814529199, 814530396, 814530412, 814530416, 814530460, 814530476, 814530480, 814530496, 814540832, 814540834, 814540840, 814540842, 814540928, 814540930, 814540931, 814540966, 814540972, 814541312, 814616581, 814616583, 814616632, 814616633, 814616635, 814616721, 814616723, 814616729, 814616731, 814617108, 814617109, 814617111, 814617117, 814617119, 814617141, 814617143, 814617288, 814617289, 814617291, 814617313, 814617315, 814617321, 814617436, 814617452, 814617456, 814617500, 814617516, 814617520, 814617536, 814617692, 814617708, 814617712, 814617756, 814617772, 814617776, 814617792, 814617856, 814618112, 814618716, 814618725, 814618727, 814618736, 814618880]))

        tile = decontamination.moc_to_healpix(
            orders,
            indices,
            int(np.log2(nside))
        )

        rms_hdu = astropy_fits.open('/Users/jodier/Downloads/EUC_MER_MOSAIC-VIS-RMS_TILE101004820-EAB536_20231026T191726.969079Z_00.00.fits')[0]
        bit_hdu = astropy_fits.open('/Users/jodier/Downloads/EUC_MER_MOSAIC-VIS-FLAG_TILE101004820-86D39F_20231026T191726.969127Z_00.00.fits')[0]

        header = rms_hdu.header

        rms_data = rms_hdu.data
        bit_data = bit_hdu.data

        if rms_data.dtype.byteorder == '>':

            rms_data.byteswap(inplace = True)
            rms_data = rms_data.newbyteorder()

        if bit_data.dtype.byteorder == '>':

            bit_data.byteswap(inplace = True)
            bit_data = bit_data.newbyteorder()

        wcs_ref = decontamination.build_healpix_wcs(astropy_WCS(header))

        t1 = time.perf_counter()
        #rms, bit, cov = decontamination.image2healpix(wcs_ref, nside, tile, rms_data, show_progress_bar = True)
        rms, bit, cov = decontamination.image_to_healpix(wcs_ref, nside, tile, rms_data, bit_data, show_progress_bar = True)
        t2 = time.perf_counter()

        delta_time = t2 - t1

        print(f'\ndelta_time: {delta_time:.4e}')

        decontamination.display_healpix(nside, tile, rms)
        plt.show()
        #decontamination.display_healpix(nside, tile, bit)
        #plt.show()
        #decontamination.display_healpix(nside, tile, cov)
        #plt.show()

########################################################################################################################

if __name__ == '__main__':

    test_pix2world()

########################################################################################################################
