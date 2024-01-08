# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np

########################################################################################################################
# JIT                                                                                                                  #
########################################################################################################################

from .jit import CPU_OPTIMIZATION_AVAILABLE, GPU_OPTIMIZATION_AVAILABLE, jit

from .jit import DeviceArray, device_array_from, device_array_empty, device_array_zeros, device_array_full

########################################################################################################################
# ALGOS                                                                                                                #
########################################################################################################################

from .algo.som_abstract import SOM_Abstract

from .algo.som_pca import SOM_PCA

from .algo.som_batch import SOM_Batch

from .algo.som_online import SOM_Online

from .algo.clustering import Clustering

########################################################################################################################
# PLOTTING                                                                                                             #
########################################################################################################################

from .plotting import catalog_to_number_density

########################################################################################################################

from .plotting.latent_space import display_latent_space

########################################################################################################################

from .plotting.healpix_map import display_healpix, display_catalog

########################################################################################################################
# GENERATORS                                                                                                           #
########################################################################################################################

from .generator.generator_uniform import Generator_Uniform

from .generator.generator_log_normal import Generator_LogNormal

from .generator.generator_number_density import Generator_NumberDensity

from .generator.generator_fullsky_uniform import Generator_FullSkyUniform

########################################################################################################################
# CORRELATIONS                                                                                                         #
########################################################################################################################

from decontamination.correlation import apodization

from decontamination.correlation.correlation_pair_count import Correlation_PairCount

from decontamination.correlation.correlation_power_spectrum import Correlation_PowerSpectrum

########################################################################################################################
# DECONTAMINATION                                                                                                      #
########################################################################################################################

from .decontamination.decontamination_som import Decontamination_SOM

from .decontamination.decontamination_elasticnet import Decontamination_ElasticNet

from .decontamination.decontamination_isd import Decontamination_ISD

########################################################################################################################
# UTILITIES                                                                                                            #
########################################################################################################################

def array_to_string(arr):

    s = np.array2string(arr, separator = ', ', suppress_small = True)

    return s.replace('[ ', '[').replace(' ]', ']')

########################################################################################################################
# EXPORTS                                                                                                              #
########################################################################################################################

__all__ = [
    'CPU_OPTIMIZATION_AVAILABLE', 'GPU_OPTIMIZATION_AVAILABLE', 'jit',
    'DeviceArray', 'device_array_from', 'device_array_empty', 'device_array_zeros', 'device_array_full',
    'SOM_Abstract', 'SOM_PCA', 'SOM_Batch', 'SOM_Online',
    'Clustering',
    'catalog_to_number_density',
    'display_latent_space',
    'display_healpix', 'display_catalog',
    'Generator_Uniform', 'Generator_LogNormal', 'Generator_NumberDensity', 'Generator_FullSkyUniform',
    'apodization', 'Correlation_PairCount', 'Correlation_PowerSpectrum',
    'Decontamination_SOM', 'Decontamination_ElasticNet', 'Decontamination_ISD',
]

########################################################################################################################
