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

    def __init__(self, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, alpha: float = 0.01, tolerance: typing.Optional[float] = None):

        ################################################################################################################

        super().__init__(dim, dtype)

        ################################################################################################################

        self._alpha = alpha
        self._tolerance = tolerance

    ####################################################################################################################

    def _update_weights(self, errors, vectors):

        m = vectors.shape[0]

        dw = -2.0 * (vectors.T @ errors) / m
        di = -2.0 * np.sum(errors) / m

        self._weights -= self._alpha * dw
        self._intercept -= self._alpha * di

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], n_epochs: typing.Optional[int] = 1000, analytic: bool = False, show_progress_bar: bool = False) -> None:

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

            X_list = []
            y_list = []

            for x, y in generator():

                X_list.append(x)
                y_list.append(y)

            X = np.vstack(X_list)
            y = np.hstack(y_list)

            ############################################################################################################

            X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

            theta = inv(X_bias.T @ X_bias) @ X_bias.T @ y

            self._weights = theta[1:]
            self._intercept = theta[0]

            ############################################################################################################

        else:

            ############################################################################################################
            # ITERATIVE METHOD                                                                                         #
            ############################################################################################################

            for epoch in tqdm.trange(n_epochs, disable = not show_progress_bar):

                ########################################################################################################

                generator = generator_builder()

                ########################################################################################################

                for vectors, y in generator():

                    errors = y - self.predict(vectors)

                    self._update_weights(errors, vectors)

                ########################################################################################################

                if self._tolerance is not None:

                    ####################################################################################################

                    weight_change = norm(self._weights - previous_weights)
                    intercept_change = abs(self._intercept - previous_intercept)

                    ####################################################################################################

                    if weight_change < self._tolerance and intercept_change < self._tolerance:

                        print(f'Convergence reached at epoch {epoch}. Stopping early.')

                        break

                    ####################################################################################################

                    previous_weights = self._weights.copy()
                    previous_intercept = self._intercept

########################################################################################################################
