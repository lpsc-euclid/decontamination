# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import typing

import numpy as np

from . import decontamination_abstract

from .isd import ISDModel

########################################################################################################################

MODEL_LINEAR_FIT = ISDModel.MODEL_LINEAR_FIT

MODEL_LINEAR_INTERP = ISDModel.MODEL_LINEAR_INTERP

########################################################################################################################

# noinspection PyPep8Naming
class Decontamination_ISD(decontamination_abstract.Decontamination_Abstract):

    ####################################################################################################################

    def __init__(self, nside: int, footprint: np.ndarray, coverage: np.ndarray, footprint_systematics: typing.Union[np.ndarray, typing.Callable], galaxy_number_density: typing.Union[np.ndarray, typing.Callable], number_of_bins: int, model: ISDModel = MODEL_LINEAR_FIT):

        ################################################################################################################
        # DATASET                                                                                                      #
        ################################################################################################################

        super().__init__(nside, footprint, coverage, footprint_systematics, galaxy_number_density)

        ################################################################################################################
        # ISD                                                                                                          #
        ################################################################################################################

        self._number_of_bins = number_of_bins

        self._model = model

        ################################################################################################################

        self._nb_iter = None

    ####################################################################################################################

    @property
    def number_of_bins(self) -> int:

        """Number of bins."""

        return self._number_of_bins

    ####################################################################################################################

    @property
    def model(self) -> ISDModel:

        """Model (see :class:`ISDModel <decontamination.decontamination.isd.ISDModel>`)."""

        return self._model

########################################################################################################################
