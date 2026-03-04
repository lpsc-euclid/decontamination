# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import enum

from .model_linear_fit import model_linear_fit

from .model_linear_interp import model_linear_interp

########################################################################################################################

class ISDModel(enum.Enum):

    MODEL_LINEAR_FIT = model_linear_fit

    MODEL_LINEAR_INTERP = model_linear_interp

########################################################################################################################
