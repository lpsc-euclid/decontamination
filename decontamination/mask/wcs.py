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

from astropy.wcs import WCS as ASTROPY_WCS

########################################################################################################################

class WCS(ASTROPY_WCS):

    """
    Thread-safe World Coordinate System (WCS) inherited from `astropy.wcs.WCS` with additional features.

    Parameters
    ----------
    astropy : bool, default: False
        ???
    thread_safe : bool, default: False
        ???
    healpix_convention : bool, default: False
        ???
    """

    ####################################################################################################################

    _MUTEX = threading.Lock()

    ####################################################################################################################

    def __init__(self, *args, astropy: bool = False, thread_safe: bool = False, healpix_convention: bool = False, **kwargs):

        ################################################################################################################

        super().__init__(*args, **kwargs)

        ################################################################################################################

        self._astropy = astropy

        self._thread_safe = thread_safe

        self._crval_rad = None

        self._cd_matrix_rad = None

        ################################################################################################################

        if healpix_convention:

            self.wcs.crval = super().all_pix2world([[
                self.wcs.crpix[0] - 0.5,
                self.wcs.crpix[1] - 0.5,
            ]], 0)[0]

        ################################################################################################################

        if len(self.wcs.ctype) == 2 and self.wcs.cunit[0] == self.wcs.cunit[1]:

            if self.wcs.cunit[0] == 'rad':

                self._crval_rad = [crval for crval in self.wcs.crval]
                self._cd_matrix_rad = self.wcs.cd

            elif self.wcs.cunit[0] == 'deg':

                self._crval_rad = [np.radians(crval) for crval in self.wcs.crval]
                self._cd_matrix_rad = np.radians(self.wcs.cd)

            elif self.wcs.cunit[0] == 'arcmin':

                self._crval_rad = [np.radians(crval / 60.0) for crval in self.wcs.crval]
                self._cd_matrix_rad = np.radians(self.wcs.cd / 60.0)

            elif self.wcs.cunit[0] == 'arcsec':

                self._crval_rad = [np.radians(crval / 3600.0) for crval in self.wcs.crval]
                self._cd_matrix_rad = np.radians(self.wcs.cd / 3600.0)

    ####################################################################################################################

    def all_pix2world(self, *args, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:

        """Thread-safe version of `astropy.wcs.WCS.all_pix2world`."""

        if self._thread_safe:

            if (not self._astropy) and (self._cd_matrix_rad is not None) and (self.wcs.ctype[0] == 'RA---TAN' and self.wcs.ctype[1] == 'DEC--TAN'):

                return self._fast_all_pix2world(*args, **kwargs)

            else:

                with WCS._MUTEX:

                    return super().all_pix2world(*args, **kwargs)

        else:

            return super().all_pix2world(*args, **kwargs)

    ####################################################################################################################

    def all_world2pix(self, *args, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:

        """Thread-safe version of `astropy.wcs.WCS.all_world2pix`."""

        if self._thread_safe:

            with WCS._MUTEX:

                return super().all_world2pix(*args, **kwargs)

        else:

            return super().all_world2pix(*args, **kwargs)

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

        if origin == 0:

            xp, yp = np.dot(self._cd_matrix_rad, np.array([
                x - (self.wcs.crpix[0] - 1.0),
                y - (self.wcs.crpix[1] - 1.0),
            ]))

        else:

            xp, yp = np.dot(self._cd_matrix_rad, np.array([
                x - (self.wcs.crpix[0] - 0.0),
                y - (self.wcs.crpix[1] - 0.0),
            ]))

        ################################################################################################################

        r = np.hypot(xp, yp)

        c = np.arctan(r)

        ################################################################################################################

        cos_crval2 = np.cos(self._crval_rad[1])
        sin_crval2 = np.sin(self._crval_rad[1])

        sin_c = np.sin(c)
        cos_c = np.cos(c)

        ################################################################################################################

        lat = np.arcsin(cos_c * sin_crval2 + np.where(r != 0.0, yp * sin_c * cos_crval2 / r, 0.0))

        lon = np.arctan2(xp * sin_c, r * cos_c * cos_crval2 - yp * sin_c * sin_crval2)

        ################################################################################################################

        return np.degrees(self._crval_rad[0] + lon), np.degrees(lat)

########################################################################################################################
