# -*- coding: utf-8 -*-
########################################################################################################################

"""
A toolbox for performing systematics decontamination in cosmology analyses.
"""

########################################################################################################################

import numpy as np

from . import jit as _jit

from .algo import pca as _pca
from .algo import som_batch as _som_batch
from .algo import som_online as _som_online

########################################################################################################################

def numpy_array_to_string_with_commas(arr):

    s = np.array2string(arr, separator = ', ', suppress_small = True)

    return s.replace('[ ', '[').replace(' ]', ']')

########################################################################################################################
# JIT                                                                                                                  #
########################################################################################################################

jit = _jit.jit

CPU_OPTIMIZATION_AVAILABLE = _jit.CPU_OPTIMIZATION_AVAILABLE
GPU_OPTIMIZATION_AVAILABLE = _jit.GPU_OPTIMIZATION_AVAILABLE

########################################################################################################################
# ALGO                                                                                                                 #
########################################################################################################################

PCA = _pca.PCA
SOMBatch = _som_batch.SOMBatch
SOMOnline = _som_online.SOMOnline

########################################################################################################################
