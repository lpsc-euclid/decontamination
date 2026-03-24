# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import typing

import numpy as np

from scipy.interpolate import interp1d

########################################################################################################################

def model_linear_interp(all_syst_centers: np.ndarray, all_syst_corrs: np.ndarray, all_systs: np.ndarray, all_syst_all_mocks_corrs: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_syst_correction_weights = []
    all_syst_delta_chi2 = []
    all_syst_params = []

    ####################################################################################################################

    for i in range(np.shape(all_syst_corrs)[0]):

        ################################################################################################################
        # COMPUTE CORRECTION                                                                                           #
        ################################################################################################################

        interp_functions = interp1d(all_syst_centers[i], all_syst_corrs[i], fill_value = 'extrapolate')

        correction_weights = 1.0 / (interp_functions(all_systs[i]))

        ################################################################################################################
        # COMPUTE χ²                                                                                                   #
        ################################################################################################################

        diff_ini = (all_syst_corrs[i] - 1).reshape(-1, 1)

        inv_cov = np.linalg.pinv(np.diag(np.diag(np.cov(all_syst_all_mocks_corrs[i]))))

        chi2_ini = (diff_ini.T @ inv_cov @ diff_ini)[0][0]

        ################################################################################################################

        all_syst_correction_weights.append(correction_weights)
        all_syst_delta_chi2.append(chi2_ini)

    ####################################################################################################################

    return (
        np.array(all_syst_correction_weights, dtype = np.float64),
        np.array(all_syst_delta_chi2, dtype = np.float64),
        np.array(all_syst_params, dtype = np.float64),
    )

########################################################################################################################
