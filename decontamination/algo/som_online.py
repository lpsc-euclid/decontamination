# -*- coding: utf-8 -*-
########################################################################################################################

import numpy as np
import numba as nb

from . import abstract_som

########################################################################################################################

class SOM_Online(abstract_som.AbstractSOM):

    def __init__(self):

        super().__init__()

########################################################################################################################
