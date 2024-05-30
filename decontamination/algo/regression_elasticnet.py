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

    def train(self, dataset: typing.Union[typing.Tuple[np.ndarray, np.ndarray], typing.Callable], n_epochs: typing.Optional[int] = 1000, fold_indices: typing.Optional[np.ndarray] = None, cv: int = 5, soft_thresholding: bool = True, compute_error: bool = True, show_progress_bar: bool = False) -> None:

        ################################################################################################################

        self._weights = np.zeros(self._dim, dtype = self._dtype)

        self._intercept = 0.0

        ################################################################################################################

        previous_weights = np.full(self._dim, np.inf, dtype = self._dtype)

        previous_intercept = np.inf

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        lambda1 = self._rho * self._l1_ratio

        lambda2 = self._rho * (1.0 - self._l1_ratio)

        for epoch in tqdm.trange(n_epochs, disable = not show_progress_bar):

            ############################################################################################################

            generator = generator_builder()

            ############################################################################################################
            # GRADIENT DESCENT METHOD                                                                                  #
            ############################################################################################################

            dw = 0.0
            di = 0.0

            n_vectors = 0

            sign_w = np.sign(self._weights)
            sign_i = np.sign(self._intercept)

            if fold_indices is None:

                ########################################################################################################
                # STANDARD GRADIENT DESCENT                                                                            #
                ########################################################################################################

                for x, y in generator():

                    n_vectors += x.shape[0]

                    errors = y - self.predict(x)

                    _dw, _di = regression_basic.Regression_Basic._update_weights(errors, x)

                    dw += _dw
                    di += _di

                ########################################################################################################

            else:

                ########################################################################################################
                # CROSS VALIDATION GRADIENT DESCENT                                                                    #
                ########################################################################################################

                for i, (x, y) in enumerate(generator()):

                    if i % cv in fold_indices:

                        n_vectors += x.shape[0]

                        errors = y - self.predict(x)

                        _dw, _di = regression_basic.Regression_Basic._update_weights(errors, x)

                        dw += _dw
                        di += _di

            ############################################################################################################

            # L2 penalty
            dw += 2.0 * lambda2 * self._weights
            di += 2.0 * lambda2 * self._intercept

            if n_vectors > 0:

                self._weights -= self._alpha * dw / n_vectors
                self._intercept -= self._alpha * di / n_vectors

            ############################################################################################################

            # L1 penalty
            if soft_thresholding:
                self._weights = np.sign(self._weights) * np.maximum(np.abs(self._weights) - self._alpha * lambda1, 0.0)
                self._intercept = np.sign(self._intercept) * np.maximum(np.abs(self._intercept) - self._alpha * lambda1, 0.0)

            else:
                self._weights -= sign_w * self._alpha * lambda1
                self._intercept -= sign_i * self._alpha * lambda1

            ############################################################################################################

            # noinspection DuplicatedCode
            if self._tolerance is not None:

                ########################################################################################################

                if norm(self._weights - previous_weights) < self._tolerance and abs(self._intercept - previous_intercept) < self._tolerance:

                    break

                ########################################################################################################

                previous_weights = self._weights.copy()
                previous_intercept = self._intercept

        ################################################################################################################

        if compute_error:

            self._error = self._compute_error(generator_builder, fold_indices = fold_indices, cv = cv)

########################################################################################################################
