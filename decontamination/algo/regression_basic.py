# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import tqdm
import typing

import numpy as np

from numpy.linalg import norm

from . import regression_abstract, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class Regression_Basic(regression_abstract.Regression_Abstract):

    """
    Basic regression.

    Parameters
    ----------
    dim : int
        Dimensionality of the input data.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]], default: **np.float32**
        Neural network data type, either **np.float32** or **np.float64**.
    """

    ####################################################################################################################

    def __init__(self, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32):

        super().__init__(dim, dtype)

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], n_epochs: typing.Optional[int] = 1000, show_progress_bar: bool = False) -> None:

        pass

########################################################################################################################
