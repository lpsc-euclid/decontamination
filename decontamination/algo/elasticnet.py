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

from decontamination.algo import dataset_to_generator_builder

########################################################################################################################

class ElasticNet(object):

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

    ####################################################################################################################

    def __init__(self, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, rho: float = 1.0, l1_ratio: float = 0.5, alpha: float = 0.01, tolerance: typing.Optional[float] = None):

        ################################################################################################################

        self._dim = dim
        self._dtype = dtype

        self._rho = rho
        self._l1_ratio = l1_ratio

        self._alpha = alpha
        self._tolerance = tolerance

        ################################################################################################################

        self._weights = None
        self._intercept = None

    ####################################################################################################################

    def _update_weights(self, errors, vectors):

        m = vectors.shape[0]

        dw = -2.0 * (vectors.T @ errors) / m \
             + self._l1_ratio * self._rho * np.sign(self._weights) \
             + 2.0 * (1.0 - self._l1_ratio) * self._rho * self._weights / m
        di = -2.0 * np.sum(errors) / m

        self._weights -= self._alpha * dw
        self._intercept -= self._alpha * di

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], n_epochs: typing.Optional[int] = 1000, show_progress_bar: bool = False) -> None:

        ################################################################################################################

        self._weights = np.zeros(self._dim, dtype = self._dtype)

        self._intercept = 0.00000000000000000000000000000000000000

        ################################################################################################################

        previous_weights = np.inf * self._weights

        previous_intercept = np.inf * self._intercept

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        for epoch in tqdm.trange(n_epochs, disable = not show_progress_bar):

            ############################################################################################################

            generator = generator_builder()

            ############################################################################################################

            for vectors, y in generator():

                errors = y - self.predict(vectors)

                self._update_weights(errors, vectors)

            ############################################################################################################

            if self._tolerance is not None:

                ############################################################################################################

                weight_change = norm(self._weights - previous_weights)
                intercept_change = abs(self._intercept - previous_intercept)

                ############################################################################################################

                if weight_change < self._tolerance and intercept_change < self._tolerance:

                    print(f'Convergence reached at epoch {epoch}. Stopping early.')

                    break

                ############################################################################################################

                previous_weights = self._weights.copy()
                previous_intercept = self._intercept

    ####################################################################################################################

    def predict(self, vectors: np.ndarray):

        return (vectors @ self._weights) + self._intercept

########################################################################################################################
