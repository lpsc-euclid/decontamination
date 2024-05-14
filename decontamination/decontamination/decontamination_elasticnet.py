# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

from . import decontamination_abstract

########################################################################################################################

# noinspection PyPep8Naming
class Decontamination_ElasticNet(decontamination_abstract.Decontamination_Abstract):

    """
    Systematics decontamination using the *Elastic Net* method.

    TODO
    """

    def __init__(self, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, beta: float = 1.0, l1: float = 0.5, alpha: float = 0.01, tolerance: float = 1e-4):

        self._dim = dim



########################################################################################################################
