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

from ..algo import regression_basic, regression_elasticnet, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class Decontamination_ElasticNet(decontamination_abstract.Decontamination_Abstract):

    """
    Systematics decontamination using the *Elastic Net* method.

    Parameters
    ----------
    dim : int
        Dimensionality of the input data.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]], default: **np.float32**
        Neural network data type, either **np.float32** or **np.float64**.
    rho : float = 1.0
        Constant that multiplies the penalty terms.
    l1_ratio : float = 0.5
        Mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    alpha : float = 0.01
        Learning rate.
    tolerance : float = **None**
        The tolerance for the optimization.
    """

    def __init__(self, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, rho: float = 1.0, l1_ratio: float = 0.5, alpha: float = 0.01, tolerance: typing.Optional[float] = None):

        ################################################################################################################
        # REGRESSION                                                                                                   #
        ################################################################################################################

        if l1_ratio == 0.0:
            self._basic = True
            self._regression = regression_basic.Regression_Basic(dim, dtype = dtype, alpha = alpha, tolerance = tolerance)
        else:
            self._basic = False
            self._regression = regression_elasticnet.Regression_ElasticNet(dim, dtype = dtype, rho = rho, l1_ratio = l1_ratio, alpha = alpha, tolerance = tolerance)

    ####################################################################################################################

    @property
    def dim(self) -> int:

        """Dimensionality of the input data."""

        return self._regression.dim

    ####################################################################################################################

    @property
    def dtype(self) -> typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]:

        """Regression data type."""

        return self._regression.dtype

    ####################################################################################################################

    def train(self, dataset: typing.Union[typing.Tuple[np.ndarray, np.ndarray], typing.Callable], n_epochs: typing.Optional[int] = 1000, soft_thresholding: bool = True, show_progress_bar: bool = False) -> None:

        if self._basic:
            self._regression.train(dataset, n_epochs = n_epochs, show_progress_bar = show_progress_bar)
        else:
            self._regression.train(dataset, n_epochs = n_epochs, soft_thresholding = soft_thresholding, show_progress_bar = show_progress_bar)

    ####################################################################################################################

    def compute_weights(self, dataset: typing.Union[np.ndarray, typing.Callable]):

        generator_builder = dataset_to_generator_builder(dataset)

        generator = generator_builder()

        for vectors, y in generator():

            yield 1.0 / (1.0 + vectors @ self._regression.weights)

########################################################################################################################
