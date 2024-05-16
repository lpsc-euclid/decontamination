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

from . import regression_basic, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class Regression_ElasticNet(regression_basic.Regression_Basic):

    """
    ElasticNet regression.

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

    __MODE__ = 'elasticnet'

    ####################################################################################################################

    def __init__(self, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, rho: float = 1.0, l1_ratio: float = 0.5, alpha: typing.Optional[float] = 0.01, tolerance: typing.Optional[float] = None):

        ################################################################################################################

        super().__init__(dim, dtype, alpha, tolerance)

        ################################################################################################################

        self._rho = rho
        self._l1_ratio = l1_ratio

    ####################################################################################################################

    def train(self, dataset: typing.Union[typing.Tuple[np.ndarray, np.ndarray], typing.Callable], n_epochs: typing.Optional[int] = 1000, soft_thresholding: bool = True, compute_error: bool = True, show_progress_bar: bool = False) -> None:

        ################################################################################################################

        self._weights = np.zeros(self._dim, dtype = self._dtype)

        self._intercept = 0.0

        ################################################################################################################

        previous_weights = np.full(self._dim, np.inf, dtype = self._dtype)

        previous_intercept = np.inf

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        l2_penality = self._rho * (1 - self._l1_ratio)

        for epoch in tqdm.trange(n_epochs, disable=not show_progress_bar):

            ############################################################################################################

            generator = generator_builder()

            ############################################################################################################

            sign = np.sign(self._weights)

            ############################################################################################################

            if self._alpha is None or self._alpha == 0.0:

                ########################################################################################################
                # COORDINATE DESCENT METHOD                                                                            #
                ########################################################################################################

                alpha = 1.0

                for x, y in generator():

                    for j in range(self._dim):

                        residual = y - (self._intercept + np.dot(x, self._weights))

                        self._weights[j] = np.dot(x[:, j], residual + x[:, j] * self._weights[j]) / (np.dot(x[:, j], x[:, j]) + l2_penality)

                    self._intercept = np.mean(y - np.dot(x, self._weights))

                ########################################################################################################

            else:

                ########################################################################################################
                # GRADIENT DESCENT METHOD                                                                              #
                ########################################################################################################

                dw = 0.0
                di = 0.0

                n_vectors = 0

                alpha = self._alpha

                for x, y in generator():

                    n_vectors += x.shape[0]

                    errors = y - self.predict(x)

                    _dw, _di = regression_basic.Regression_Basic._update_weights(errors, x)

                    dw += _dw
                    di += _di

                ########################################################################################################

                # L2 penalty
                dw += 2.0 * l2_penality * self._weights

                self._weights -= alpha * dw / n_vectors
                self._intercept -= alpha * di / n_vectors

            ############################################################################################################

            # L1 penalty
            if soft_thresholding:
                self._weights = np.sign(self._weights) * np.maximum(np.abs(self._weights) - alpha * self._l1_ratio * self._rho, 0.0)
            else:
                self._weights -= sign * alpha * self._l1_ratio * self._rho

            ############################################################################################################

            # noinspection DuplicatedCode
            if self._tolerance is not None:

                ########################################################################################################

                weight_change = norm(self._weights - previous_weights)
                intercept_change = abs(self._intercept - previous_intercept)

                ########################################################################################################

                if weight_change < self._tolerance and intercept_change < self._tolerance:

                    print(f'Convergence reached at epoch {epoch}. Stopping early.')

                    break

                ########################################################################################################

                previous_weights = self._weights.copy()
                previous_intercept = self._intercept

        ################################################################################################################

        if compute_error:

            self._error = self._compute_error(generator_builder)

########################################################################################################################
