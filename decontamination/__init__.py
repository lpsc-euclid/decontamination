# -*- coding: utf-8 -*-
########################################################################################################################

"""
.. include:: ../docs/header.md

A toolbox for performing systematics decontamination in cosmology analyses.
"""

########################################################################################################################

import numpy as np

########################################################################################################################
# JIT                                                                                                                  #
########################################################################################################################

from ._jit import CPU_OPTIMIZATION_AVAILABLE, GPU_OPTIMIZATION_AVAILABLE, jit

from ._jit import DeviceArray, device_array_from, device_array_empty, device_array_zeros, device_array_full

########################################################################################################################
# ALGO                                                                                                                 #
########################################################################################################################

from ._algo.som_abstract import SOM_Abstract

from ._algo.som_pca import SOM_PCA

from ._algo.som_batch import SOM_Batch

from ._algo.som_online import SOM_Online

from ._algo.clustering import Clustering

########################################################################################################################
# PLOTTING                                                                                                             #
########################################################################################################################

from ._plotting.latent_space import display_latent_space
from ._plotting.clustering import display_clusters

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
    'display_latent_space', 'display_clusters'
]

########################################################################################################################
