# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

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
    def weights(self):

        return self._weights

    ####################################################################################################################

    @property
    def intercept(self):

        return self._intercept

    ####################################################################################################################

    def predict(self, dataset: np.ndarray):

        return (dataset @ self._weights) + self._intercept

    ####################################################################################################################

    def predict_generator(self, dataset: typing.Union[np.ndarray, typing.Callable]):

        generator_builder = dataset_to_generator_builder(dataset)

        generator = generator_builder()

        for vectors, y in generator():

            yield (vectors @ self._weights) + self._intercept

########################################################################################################################
