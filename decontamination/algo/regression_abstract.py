# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import math
import typing

import numpy as np

from . import dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class Regression_Abstract(object):

    """
    ElasticNet regression.

    Parameters
    ----------
    dim : int
        Dimensionality of the input data.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]], default: **np.float32**
        Neural network data type, either **np.float32** or **np.float64**.
    """

    __MODE__ = 'abstract'

    ####################################################################################################################

    def __init__(self, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32):

        ################################################################################################################

        self._dim = dim
        self._dtype = dtype

        ################################################################################################################

        self._weights = np.zeros(self._dim, dtype = self._dtype)

        self._intercept = 0.00000000000000000000000000000000000000

        ################################################################################################################

        self._error = None

    ####################################################################################################################

    @property
    def dim(self) -> int:

        """Dimensionality of the input data."""

        return self._dim

    ####################################################################################################################

    @property
    def dtype(self) -> typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]:

        """Regression data type."""

        return self._dtype

    ####################################################################################################################

    @property
    def weights(self) -> np.ndarray:

        return self._weights

    ####################################################################################################################

    @property
    def intercept(self) -> float:

        return self._intercept

    ####################################################################################################################

    @property
    def error(self) -> float:

        return self._error

    ####################################################################################################################

    def predict(self, x: np.ndarray) -> np.ndarray:

        return x @ self._weights + self._intercept

    ####################################################################################################################

    def predict_generator(self, dataset: typing.Union[np.ndarray, typing.Callable]):

        generator_builder = dataset_to_generator_builder(dataset)

        generator = generator_builder()

        for x, y in generator():

            yield x @ self._weights + self._intercept

    ####################################################################################################################

    def _compute_error(self, generator_builder: typing.Callable, fold_indices: typing.Optional[typing.List[int]] = None, cv: int = 5):

        result = 0.0

        n_vectors = 0

        ################################################################################################################

        generator = generator_builder()

        ################################################################################################################

        if fold_indices is None:

            ############################################################################################################
            # STANDARD ERROR                                                                                           #
            ############################################################################################################

            for x, y in generator():

                n_vectors += x.shape[0]

                result += np.sum((y - self.predict(x)) ** 2)

            ############################################################################################################

        else:

            ############################################################################################################
            # CROSS VALIDATION ERROR                                                                                   #
            ############################################################################################################

            for i, (x, y) in enumerate(generator()):

                if i % cv not in fold_indices:

                    n_vectors += x.shape[0]

                    result += np.sum((y - self.predict(x)) ** 2)

        ################################################################################################################

        return math.sqrt(result / n_vectors) if n_vectors > 0 else math.inf

########################################################################################################################
