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

    """Thread-safe and HEALPix compliant World Coordinate System (WCS)"""

    ####################################################################################################################

    def __init__(self, *args, healpix_convention: bool = True, **kwargs):

        """???"""

        ################################################################################################################

        super().__init__(*args, **kwargs)

        ################################################################################################################

        self._mutex = threading.Lock()

        ################################################################################################################

        if healpix_convention:

            # On HEALPix, we want the value at the pixel center.

            self.wcs.crval = self.all_pix2world([[self.wcs.crpix[0] - 0.5, self.wcs.crpix[1] - 0.5]], 0)[0]

    ####################################################################################################################

    def all_pix2world(self, *args, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:

        """???"""

        with self._mutex:

            return super().all_pix2world(*args, **kwargs)

    ####################################################################################################################

    def all_world2pix(self, *args, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:

        """???"""

        with self._mutex:

            return super().all_world2pix(*args, **kwargs)

########################################################################################################################
