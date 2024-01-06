# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np
import healpy as hp

########################################################################################################################

def apodization(full_sky_footprint, fwhm, threshold = 1.0e-5, nest : bool = True):

    """
    Applies Gaussian smoothing to a full-sky map for edge minimization (= apodizedization).

    Parameters
    ----------
    full_sky_footprint : np.ndarray
        The input full-sky footprint to be apodized.
    fwhm : float
        The full width half max parameter of the Gaussian (in arcmins).
    threshold : float, default: 1.0e-5
        Sets lower and upper value limits to mitigate extreme values after smoothing.
    nest : bool, default: True
        If **True**, assumes NESTED pixel ordering, otherwise, RING pixel ordering (default: **True**).

    Returns
    -------
    np.ndarray
        Apodized version of the input full-sky footprint.
    """

    result = hp.smoothing(full_sky_footprint, fwhm = np.deg2rad(fwhm / 60.0), pol = False)

    result[result < (0.0 + threshold)] = 0.0 + threshold
    result[result > (1.0 - threshold)] = 1.0 - threshold

    return result

########################################################################################################################
