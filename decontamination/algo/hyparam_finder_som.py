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

from . import som_pca, som_batch, som_online

########################################################################################################################

class HypParamFinder_SOM(object):

    ####################################################################################################################

    ALPHA_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]

    ####################################################################################################################

    def __init__(self, dataset: np.ndarray, dataset_weights: np.ndarray, topology: str, batch: bool = True, use_best_epoch: bool = True, show_progress_bar: bool = True, enable_gpu: bool = True, threads_per_blocks: typing.Optional[int] = None):

        self._dataset = dataset
        self._dataset_weights = dataset_weights

        self._topology = topology
        self._batch = batch

        self._use_best_epoch = use_best_epoch
        self._show_progress_bar = show_progress_bar
        self._enable_gpu = enable_gpu
        self._threads_per_blocks = threads_per_blocks

    ####################################################################################################################

    def find(self, n_epochs):

        ################################################################################################################

        result_m = None
        result_sigma = None
        result_alpha = None

        min_qe = float('inf')

        ################################################################################################################

        max_m = np.sqrt(5.0 * np.sqrt(self._dataset.shape[0]))

        ################################################################################################################

        m_list = np.linspace(0.5 * max_m, 2.0 * max_m, num = 7, dtype = float)

        for m in tqdm.tqdm(m_list, disable = not self._show_progress_bar):

            sigma_list = m / np.linspace(2.0, 4.0, num = 5, dtype = float)

            for sigma in sigma_list:

                for alpha in ([None] if self._batch else self.ALPHA_LIST):

                    qe = self._train(m, alpha, sigma, n_epochs)

                    if min_qe > qe:

                        min_qe = qe

                        result_m = m
                        result_alpha = alpha
                        result_sigma = sigma

        ################################################################################################################

        return result_m, result_alpha, result_sigma

    ####################################################################################################################

    def _train(self, m: int, alpha: typing.Optional[float], sigma: float, n_epochs: int):

        ################################################################################################################
        # PCA                                                                                                          #
        ################################################################################################################

        pca = som_pca.SOM_PCA(m, m, self._dataset.shape[1], dtype = self._dataset.dtype, topology = self._topology)

        ################################################################################################################

        pca.train(self._dataset, dataset_weights = self._dataset_weights, min_weight = 0.0, max_weight = 1.0)

        ################################################################################################################
        # SOM                                                                                                          #
        ################################################################################################################

        if self._batch:
            som = som_batch.SOM_Batch(m, m, self._dataset.shape[1], dtype = self._dataset.dtype, topology = self._topology, sigma = sigma)
        else:
            som = som_online.SOM_Online(m, m, self._dataset.shape[1], dtype = self._dataset.dtype, topology = self._topology, alpha = alpha, sigma = sigma)

        ################################################################################################################

        som.init_from(pca)

        ################################################################################################################

        if self._batch:
            som.train(self._dataset, dataset_weights = self._dataset_weights, n_epochs = n_epochs, n_vectors = None, use_best_epoch = self._use_best_epoch, show_progress_bar = self._show_progress_bar, enable_gpu = self._enable_gpu, threads_per_blocks = self._threads_per_blocks)
        else:
            som.train(self._dataset, dataset_weights = self._dataset_weights, n_epochs = n_epochs, n_vectors = None, use_best_epoch = self._use_best_epoch, show_progress_bar = self._show_progress_bar)

        ################################################################################################################
        # GET QE                                                                                                       #
        ################################################################################################################

        return np.min(som.quantization_errors) if self._use_best_epoch else som.quantization_errors[-1]

########################################################################################################################
