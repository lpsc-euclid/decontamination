# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import numpy as np
import healpy as hp

########################################################################################################################

def apodization(full_sky_map: np.ndarray, fwhm: float, threshold: float = 1.0e-5, nest: bool = True) -> np.ndarray:

    """
    Applies Gaussian smoothing to a full-sky map for edge minimization (= apodization).

    Parameters
    ----------
    full_sky_map : np.ndarray
        The input full-sky map to be apodized.
    fwhm : float
        The full width half max parameter of the Gaussian (in arcmins).
    threshold : float, default: **1.0e-5**
        Sets lower and upper limits to mitigate extreme values after smoothing.
    nest : bool, default: **True**
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering.

    Returns
    -------
    np.ndarray
        Apodized version of the input full-sky map.

    :private:
    """

    result = hp.smoothing(full_sky_map, fwhm = np.deg2rad(fwhm / 60.0), pol = False, nest = nest)

    result[result < (0.0 + threshold)] = 0.0 + threshold
    result[result > (1.0 - threshold)] = 1.0 - threshold

    return result

########################################################################################################################
