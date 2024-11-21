# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import numpy as np

########################################################################################################################
# JIT                                                                                                                  #
########################################################################################################################

from .jit import CPU_OPTIMIZATION_AVAILABLE, GPU_OPTIMIZATION_AVAILABLE, jit

from .jit import get_max_cpu_cores, get_max_gpu_threads, get_num_cpu_threads, set_num_cpu_threads

from .jit import DeviceArray, device_array_from, device_array_empty, device_array_zeros, device_array_full

########################################################################################################################
# HP                                                                                                                   #
########################################################################################################################

from .hp import nside2npix, npix2nside, nside2pixarea, nside2resol, xyf2nest, nest2xyf, ang2pix, randang

########################################################################################################################
# MASK                                                                                                                 #
########################################################################################################################

from .mask.wcs import WCS

from .mask.ud_grade import ud_grade

from .mask.healpix import rms_bit_to_healpix, image_to_healpix

from .mask.moc import order_index_to_nuniq, nuniq_to_order_index, moc_to_healpix, wmoc_to_healpix

########################################################################################################################
# ALGOS                                                                                                                #
########################################################################################################################

from .algo.som_abstract import SOM_Abstract

from .algo.som_pca import SOM_PCA

from .algo.som_batch import SOM_Batch

from .algo.som_online import SOM_Online

from .algo.regression_abstract import Regression_Abstract

from .algo.regression_basic import Regression_Basic

from .algo.clustering import Clustering

from .algo.selection import Selection

########################################################################################################################
# PLOTTING                                                                                                             #
########################################################################################################################

from .plotting import get_bounding_box, get_full_sky, catalog_to_number_density

########################################################################################################################

from .plotting.latent_space import display_latent_space

########################################################################################################################

from .plotting.healpix_map import display_healpix, display_catalog

########################################################################################################################
# GENERATORS                                                                                                           #
########################################################################################################################

from .generator.generator_uniform import Generator_Uniform

from .generator.generator_number_density import Generator_NumberDensity

from .generator.generator_fullsky_uniform import Generator_FullSkyUniform

########################################################################################################################
# CORRELATIONS                                                                                                         #
########################################################################################################################

from decontamination.correlation.correlation_abstract import Correlation_Abstract

from decontamination.correlation.correlation_pair_count import Correlation_PairCount

from decontamination.correlation.correlation_scalar import Correlation_Scalar

from decontamination.correlation.correlation_scalar_alt import Correlation_ScalarAlt

########################################################################################################################
# DECONTAMINATION                                                                                                      #
########################################################################################################################

from .decontamination.decontamination_abstract import Decontamination_Abstract

from .decontamination.decontamination_som import Decontamination_SOM

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
    'get_max_cpu_cores', 'get_max_gpu_threads', 'get_num_cpu_threads', 'set_num_cpu_threads',
    'DeviceArray', 'device_array_from', 'device_array_empty', 'device_array_zeros', 'device_array_full',
    'nside2npix', 'npix2nside', 'nside2pixarea', 'nside2resol', 'xyf2nest', 'nest2xyf', 'ang2pix', 'randang',
    'WCS', 'ud_grade', 'rms_bit_to_healpix', 'image_to_healpix',
    'order_index_to_nuniq', 'nuniq_to_order_index', 'moc_to_healpix', 'wmoc_to_healpix',
    'SOM_Abstract', 'SOM_PCA', 'SOM_Batch', 'SOM_Online',
    'Regression_Abstract', 'Regression_Basic',
    'Clustering',
    'Selection',
    'get_bounding_box', 'get_full_sky', 'catalog_to_number_density',
    'display_latent_space',
    'display_healpix', 'display_catalog',
    'Generator_Uniform', 'Generator_NumberDensity', 'Generator_FullSkyUniform',
    'Correlation_Abstract', 'Correlation_PairCount', 'Correlation_Scalar',
    'Decontamination_Abstract', 'Decontamination_SOM',
]

########################################################################################################################
