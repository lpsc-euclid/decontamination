# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import typing
import threading

import numpy as np

from astropy.wcs import WCS as AstropyWCS

########################################################################################################################

class WCS:

    """Thread-safe and HEALPix compliant World Coordinate System (WCS)"""

    ####################################################################################################################

    def __init__(self, ctype1: str, ctype2: str, cunit1: str, cunit2: str, crpix1: int, crpix2: int, crval1: float, crval2: float, cd1_1: float, cd1_2: float, cd2_1: float, cd2_2: float, healpix_convention: bool = True):

        """???"""

        ################################################################################################################

        self._mutex = threading.Lock()

        ################################################################################################################

        self._astropy_wcs = AstropyWCS(naxis = 2)

        self._astropy_wcs.wcs.ctype = [ctype1, ctype2]
        self._astropy_wcs.wcs.cunit = [cunit1, cunit2]
        self._astropy_wcs.wcs.crpix = [crpix1, crpix2]
        self._astropy_wcs.wcs.crval = [crval1, crval2]

        self._astropy_wcs.wcs.cd = np.array([[cd1_1, cd1_2], [cd2_1, cd2_2]], dtype = np.float64)

        ################################################################################################################

        if healpix_convention:

            # On HEALPix, we want the value at the pixel center.

            ############################################################################################################

            v = np.array([[
                self._astropy_wcs.wcs.crpix[0] - 0.5,
                self._astropy_wcs.wcs.crpix[1] - 0.5,
            ]], dtype = self._astropy_wcs.wcs.crval.dtype)

            ############################################################################################################

            self._astropy_wcs.wcs.crval = self._astropy_wcs.all_pix2world(v, 0)[0]

    ####################################################################################################################

    @staticmethod
    def from_fits_header(header, healpix_convention: bool = True) -> 'WCS':

        """???"""

        return WCS(
            header['CTYPE1'], header['CTYPE2'],
            header['CUNIT1'], header['CUNIT2'],
            header['CRPIX1'], header['CRPIX2'],
            header['CRVAL1'], header['CRVAL2'],
            header['CD1_1'], header['CD1_2'],
            header['CD2_1'], header['CD2_2'],
            healpix_convention = healpix_convention
        )

    ####################################################################################################################

    @property
    def wcs(self):

        """???"""

        return self._astropy_wcs.wcs

    ####################################################################################################################

    def all_pix2world(self, x: np.ndarray, y: np.ndarray, ra_dec_order: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:

        """???"""

        with self._mutex:

            return self._astropy_wcs.all_pix2world(x, y, 0, ra_dec_order = ra_dec_order)

    ####################################################################################################################

    def all_world2pix(self, θ: np.ndarray, ϕ: np.ndarray, ra_dec_order: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:

        """???"""

        with self._mutex:

            return self._astropy_wcs.all_world2pix(θ, ϕ, 0, ra_dec_order = ra_dec_order)

########################################################################################################################
