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

class WCS(AstropyWCS):

    """Thread-safe and HEALPix compliant World Coordinate System (WCS) inherited from `astropy.wcs.WCS`."""

    ####################################################################################################################

    def __init__(self, *args, astropy: bool = False, thread_safe: bool = False, healpix_convention: bool = True, **kwargs):

        ################################################################################################################

        super().__init__(*args, **kwargs)

        ################################################################################################################

        self._mutex = threading.Lock()

        ################################################################################################################

        self._astropy = astropy

        self._thread_safe = thread_safe

        ################################################################################################################

        if healpix_convention:

            self.wcs.crval = super().all_pix2world([[
                self.wcs.crpix[0] - 0.5,
                self.wcs.crpix[1] - 0.5,
            ]], 0)[0]

    ####################################################################################################################

    def all_pix2world(self, *args, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:

        """Thread-safe version of `astropy.wcs.WCS.all_pix2world`."""

        if self._thread_safe:

            if not self._astropy and len(self.wcs.ctype) == 2 and self.wcs.ctype[0] == 'RA---TAN' and self.wcs.ctype[1] == 'DEC--TAN' and self.wcs.cunit[0] == self.wcs.cunit[1]:

                return self._fast_all_pix2world(*args, **kwargs)

            else:

                with self._mutex:

                    return super().all_pix2world(*args, **kwargs)

        else:

            return super().all_pix2world(*args, **kwargs)

    ####################################################################################################################

    def all_world2pix(self, *args, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:

        """Thread-safe version of `astropy.wcs.WCS.all_world2pix`."""

        if self._thread_safe:

            with self._mutex:

                return super().all_world2pix(*args, **kwargs)

        else:

            return super().all_world2pix(*args, **kwargs)

    ####################################################################################################################
    ####################################################################################################################

    def _fast_all_pix2world(self, *args, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:

        ################################################################################################################
        # GET PARAMETERS                                                                                               #
        ################################################################################################################

        l = len(args)

        if l == 2:

            x, y = np.asarray(args[0]).T

            origin = args[1]

        elif l == 3:

            x, y = args[0], args[1]

            origin = args[2]

        else:

            raise ValueError('Invalid number of arguments')

        ################################################################################################################
        # COMPUTE ANGLES                                                                                               #
        ################################################################################################################

        if self.wcs.cunit[0] == 'rad':

            crval_rad = [crval for crval in self.wcs.crval]
            cd_rad = self.wcs.cd

        elif self.wcs.cunit[0] == 'deg':

            crval_rad = [np.radians(crval) for crval in self.wcs.crval]
            cd_rad = np.radians(self.wcs.cd)

        elif self.wcs.cunit[0] == 'arcmin':

            crval_rad = [np.radians(crval / 60.0) for crval in self.wcs.crval]
            cd_rad = np.radians(self.wcs.cd / 60.0)

        elif self.wcs.cunit[0] == 'arcsec':

            crval_rad = [np.radians(crval / 3600.0) for crval in self.wcs.crval]
            cd_rad = np.radians(self.wcs.cd / 3600.0)

        else:

            raise ValueError('Unsupported CUNIT configuration')

        ################################################################################################################

        if origin == 0:

            xp, yp = np.dot(cd_rad, np.array([
                x - (self.wcs.crpix[0] - 1.0),
                y - (self.wcs.crpix[1] - 1.0),
            ]))

        else:

            xp, yp = np.dot(cd_rad, np.array([
                x - (self.wcs.crpix[0] - 0.0),
                y - (self.wcs.crpix[1] - 0.0),
            ]))

        ################################################################################################################

        r = np.hypot(xp, yp)

        c = np.arctan(r)

        ################################################################################################################

        cos_crval2 = np.cos(crval_rad[1])
        sin_crval2 = np.sin(crval_rad[1])

        sin_c = np.sin(c)
        cos_c = np.cos(c)

        ################################################################################################################

        lat = np.arcsin(cos_c * sin_crval2 + (yp * sin_c * cos_crval2 / r))

        lon = crval_rad[0] + np.arctan2(xp * sin_c, r * cos_crval2 * cos_c - yp * sin_crval2 * sin_c)

        ################################################################################################################

        center_mask = np.where(r == 0.0)[0]

        lon[center_mask] = crval_rad[1]
        lat[center_mask] = crval_rad[0]

        ################################################################################################################

        return np.degrees(lon), np.degrees(lat)

########################################################################################################################
