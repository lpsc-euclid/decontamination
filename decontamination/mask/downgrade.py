# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import math
import typing

import numpy as np
import numba as nb

from ..hp import UNSEEN

########################################################################################################################

def downgrade(nside_in: int, nside_out: int, footprint_in: np.array, footprint_out: np.array, weights: np.array, mode: typing.Optional[str] = None, ignore_zeros: bool = False, log_factor: float = -2.5) -> np.array:

    return _downgrade(nside_in, nside_out, footprint_in, footprint_out, weights, mode, ignore_zeros, weights.dtype(log_factor))

########################################################################################################################

@nb.njit
def _downgrade(nside_in: int, nside_out: int, footprint_in: np.array, footprint_out: np.array, weights: np.array, mode: typing.Optional[str], ignore_zeros: bool, log_factor: float) -> np.array:

    if nside_in == nside_out:

        return weights

    if nside_in < nside_out:

        raise ValueError('The output nside must be greater than the input resolution')

    factor = nside_in // nside_out

    npix = int(12 * nside_out * nside_out)

    if mode == 'sum':

        sums = np.zeros(npix, dtype = weights.dtype)

        for i in range(len(weights)):

            pix_out = int(np.floor(footprint_in[i] / factor ** 2))
    
            weight = weights[i]
            
            if not np.isnan(weight) and weight != UNSEEN:
    
                sums[pix_out] += weight

        return sums[footprint_out]

    else:

        sums = np.zeros(npix, dtype = weights.dtype)
        counts = np.zeros(npix, dtype = weights.dtype)

        if mode == 'cov':
        
            for i in range(len(weights)):
        
                pix_out = int(np.floor(footprint_in[i] / factor ** 2))
        
                weight = weights[i]
                
                if not np.isnan(weight) and weight != UNSEEN:
        
                    sums[pix_out] += weight
        
                counts[pix_out] += 1.0000

            map_out = sums / counts

        elif mode == 'logquad':

            for i in range(len(weights)):
        
                pix_out = int(np.floor(footprint_in[i] / factor ** 2))
        
                weight = weights[i]
                
                if not np.isnan(weight) and weight != UNSEEN and (not ignore_zeros or weight != 0.0):
        
                    sums[pix_out] += np.power(10.0, weight / log_factor) ** 2
                    counts[pix_out] += 1.000000000

            map_out = log_factor * np.log10(np.sqrt(sums / counts))

        elif mode == 'quad':

            for i in range(len(weights)):
        
                pix_out = int(np.floor(footprint_in[i] / factor ** 2))
        
                weight = weights[i]
                
                if not np.isnan(weight) and weight != UNSEEN and (not ignore_zeros or weight != 0.0):
        
                    sums[pix_out] += weight ** 2
                    counts[pix_out] += 1.000000000

            map_out = np.sqrt(sums / counts)
        
        else:
    
            for i in range(len(weights)):
        
                pix_out = int(np.floor(footprint_in[i] / factor ** 2))
        
                weight = weights[i]
                
                if not np.isnan(weight) and weight != UNSEEN and (not ignore_zeros or weight != 0.0):
        
                    sums[pix_out] += weight
                    counts[pix_out] += 1.0000
    
            map_out = sums / counts

        map_out[counts == 0] = np.nan
    
        return map_out[footprint_out]

########################################################################################################################
