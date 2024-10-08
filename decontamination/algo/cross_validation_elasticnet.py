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

from . import regression_elasticnet, dataset_to_generator_builder

########################################################################################################################

try:

    from sklearn.linear_model import ElasticNet as SklearnElastic

except (ImportError, ModuleNotFoundError):

    SklearnElastic = None

########################################################################################################################

RESOLUTION = np.finfo(float).resolution

########################################################################################################################

# noinspection PyPep8Naming
class CrossValidation_ElasticNet(regression_elasticnet.Regression_ElasticNet):

    __MODE__ = 'elasticnetcv'

    ####################################################################################################################

    def __init__(self, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, l1_ratios: typing.Union[float, typing.List[float]] = 0.5, n_rhos: int = 100, eps: float = 1.0e-3, cv: int = 5, alpha: typing.Optional[float] = 0.01, tolerance: typing.Optional[float] = None):

        ################################################################################################################

        super().__init__(dim, dtype, rho = 0.0, l1_ratio = 0.0, alpha = alpha, tolerance = tolerance)

        ################################################################################################################

        if isinstance(l1_ratios, float) or isinstance(l1_ratios, int):

            l1_ratios = [l1_ratios]

        ################################################################################################################

        self._l1_ratios = l1_ratios
        self._n_rhos = n_rhos
        self._eps = eps
        self._cv = cv

    ####################################################################################################################

    def _compute_rho_range(self, generator_builder: typing.Callable, lambda_max: np.ndarray, l1_ratio: float) -> typing.Tuple[np.ndarray, np.ndarray]:

        ################################################################################################################

        if lambda_max is None:

            ############################################################################################################

            n_vectors = 0

            xy = np.zeros(self._dim, dtype = np.float32)

            ############################################################################################################

            generator = generator_builder()

            for x, y in generator():

                n_vectors += x.shape[0]

                xy += np.dot(x.T, y)

            ############################################################################################################

            lambda_max = np.sqrt(xy ** 2).max() / n_vectors

        ################################################################################################################

        alpha_max = lambda_max / l1_ratio

        ################################################################################################################

        if alpha_max <= RESOLUTION:

            return lambda_max, np.full(self._n_rhos, RESOLUTION)

        else:

            return lambda_max, np.geomspace(alpha_max, alpha_max * self._eps, num = self._n_rhos)

    ####################################################################################################################

    def find_hyper_parameters(self, dataset: typing.Union[typing.Tuple[np.ndarray, np.ndarray], typing.Callable], n_epochs: typing.Optional[int] = 1000, soft_thresholding: bool = True, seed: typing.Optional[int] = None, use_sklearn: bool = False, show_progress_bar: bool = False) -> typing.Optional[typing.Dict[str, float]]:

        result = None

        lambda_max = None

        best_score = np.inf

        ################################################################################################################

        rng = np.random.default_rng(seed = seed)

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        for l1_ratio in self._l1_ratios:

            print(f'-> l1_ratio = {l1_ratio}')

            ############################################################################################################
            # COMPUTE RHO RANGE                                                                                        #
            ############################################################################################################

            lambda_max, rhos = self._compute_rho_range(generator_builder, lambda_max, l1_ratio)

            print('rhos ->', rhos)

            ############################################################################################################
            # COMPUTE FOLDS                                                                                            #
            ############################################################################################################

            indices = np.arange(self._cv)

            rng.shuffle(indices)

            folds = [indices[i::self._cv] for i in range(self._cv)]

            ############################################################################################################
            # FIND BEST HYPER PARAMETERS                                                                               #
            ############################################################################################################

            for rho in tqdm.tqdm(rhos, disable = not show_progress_bar):

                ########################################################################################################

                scores = []

                for i in range(self._cv):

                    ####################################################################################################

                    fold_indices = np.concatenate([folds[j] for j in range(self._cv) if j != i])

                    print(fold_indices)

                    ####################################################################################################

                    if use_sklearn:

                        ################################################################################################

                        generator = generator_builder()

                        dataset_x = np.empty((0, self._dim), dtype = self._dtype)
                        dataset_y = np.empty(0, dtype = self._dtype)

                        for j, (x, y) in enumerate(generator()):

                            if j % self._cv in fold_indices:

                                dataset_x = np.append(dataset_x, x, axis = 0)
                                dataset_y = np.append(dataset_y, y, axis = 0)

                        ################################################################################################

                        enet = SklearnElastic(alpha = rho, l1_ratio = l1_ratio, fit_intercept = True, max_iter = n_epochs, tol = self._tolerance)

                        enet.fit(dataset_x, dataset_y)

                        ################################################################################################

                        self._weights = enet.coef_.copy().astype(self.dtype)

                        self._intercept = self.dtype(enet.intercept_)

                        ################################################################################################

                    else:

                        ################################################################################################

                        self._rho = rho
                        self._l1_ratio = l1_ratio

                        self.train(dataset, n_epochs = n_epochs, fold_indices = fold_indices, cv = self._cv, soft_thresholding = soft_thresholding, compute_error = False, show_progress_bar = False)

                ########################################################################################################

                scores.append(self._compute_error(generator_builder, fold_indices = fold_indices, cv = self._cv))

                ########################################################################################################

                mean_score = np.mean(scores)

                ########################################################################################################

                if mean_score < best_score:

                    best_score = mean_score

                    result = {
                        'rho': rho,
                        'l1_ratio': l1_ratio,
                    }

        ################################################################################################################

        if result is not None:

            self._rho = result['rho']
            self._l1_ratio = result['l1_ratio']

        ################################################################################################################

        return result

    ####################################################################################################################

    def set_hyper_parameters(self, rho, l1_ratio) -> None:

        self._rho = rho
        self._l1_ratio = l1_ratio

########################################################################################################################
