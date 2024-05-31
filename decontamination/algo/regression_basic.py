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
import numba as nb

from numpy.linalg import inv, norm

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
    alpha : float = 0.01
        Learning rate.
    tolerance : float = **None**
        The tolerance for the optimization.
    """

    __MODE__ = 'basic'

    ####################################################################################################################

    def __init__(self, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, alpha: typing.Optional[float] = 0.01, tolerance: typing.Optional[float] = None):

        ################################################################################################################

        super().__init__(dim, dtype)

        ################################################################################################################

        self._alpha = alpha or 0.01
        self._tolerance = tolerance

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _update_weights(weights: np.ndarray, intercept: float, x: np.ndarray, y: np.ndarray) -> typing.Tuple[np.ndarray, float]:

        errors = y - (x @ weights + intercept)

        dw = -2.0 * (x.T @ errors)

        di = -2.0 * np.sum(errors)

        return dw, di

    ####################################################################################################################

    def train(self, dataset: typing.Union[typing.Tuple[np.ndarray, np.ndarray], typing.Callable], n_epochs: typing.Optional[int] = 1000, analytic: bool = True, compute_error: bool = True, show_progress_bar: bool = False) -> None:

        ################################################################################################################

        self._weights = np.zeros(self._dim, dtype = self._dtype)

        self._intercept = 0.0

        ################################################################################################################

        previous_weights = np.full(self._dim, np.inf, dtype = self._dtype)

        previous_intercept = np.inf

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        if analytic:

            ############################################################################################################
            # ANALYTIC METHOD                                                                                          #
            ############################################################################################################

            generator = generator_builder()

            ############################################################################################################

            xtx = None
            xty = None

            for x, y in generator():

                x_bias = np.hstack((np.ones((x.shape[0], 1), dtype = self._dtype), x))

                if xtx is None:
                    xtx = x_bias.T @ x_bias
                    xty = x_bias.T @ y
                else:
                    xtx += x_bias.T @ x_bias
                    xty += x_bias.T @ y

            ############################################################################################################

            theta = inv(xtx) @ xty

            self._weights = theta[1:]
            self._intercept = theta[0]

            ############################################################################################################

        else:

            ############################################################################################################
            # ITERATIVE METHODS                                                                                        #
            ############################################################################################################

            for _ in tqdm.trange(n_epochs, disable = not show_progress_bar):

                ########################################################################################################

                generator = generator_builder()

                ########################################################################################################
                # GRADIENT DESCENT METHOD                                                                              #
                ########################################################################################################

                dw = 0.0
                di = 0.0

                n_vectors = 0

                for x, y in generator():

                    n_vectors += x.shape[0]

                    _dw, _di = Regression_Basic._update_weights(self._weights, self._dtype(self._intercept), x, y)

                    dw += _dw
                    di += _di

                ########################################################################################################

                self._weights -= self._alpha * dw / (2.0 * n_vectors)
                self._intercept -= self._alpha * di / (2.0 * n_vectors)

                ########################################################################################################

                # noinspection DuplicatedCode
                if self._tolerance is not None:

                    ####################################################################################################

                    if norm(self._weights - previous_weights) < self._tolerance and abs(self._intercept - previous_intercept) < self._tolerance:

                        break

                    ####################################################################################################

                    previous_weights = self._weights.copy()
                    previous_intercept = self._intercept

        ################################################################################################################

        if compute_error:

            self._error = self._compute_error(generator_builder)

########################################################################################################################
