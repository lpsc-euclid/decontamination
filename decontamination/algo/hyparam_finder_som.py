# -*- coding: utf-8 -*-
########################################################################################################################
# author: Jérôme ODIER <jerome.odier@lpsc.in2p3.fr>
#         Gaël ALGUERO <gael.alguero@lpsc.in2p3.fr>
#         Juan MACIAS-PEREZ <juan.macias-perez@lpsc.in2p3.fr>
# license: CeCILL-C
########################################################################################################################

import math
import tqdm
import typing

import numpy as np

from . import som_pca, som_batch, som_online, dataset_to_generator_builder

########################################################################################################################

class HypParamFinder_SOM(object):

    """
    Estimates the best hyperparameters (giving the best quantization error) using a grid.

    Parameters
    ----------
    dataset : typing.Union[np.ndarray, typing.Callable]
        Training dataset array or generator builder.
    dataset_weights : typing.Union[np.ndarray, typing.Callable], default: **None**
        Training dataset weight array or generator builder.
    topology : str, default: **None** ≡ **'hexagonal'**
        Neural network topology, either **'square'** or **'hexagonal'**.
    batch : bool, default: **True**
        ???
    use_best_epoch : bool, default: **True**
        ???
    show_progress_bar : bool, default: **False**
        Specifies whether to display a progress bar.
    enable_gpu : bool, default: **True**
        If available, run on GPU rather than CPU.
    threads_per_blocks : int, default: **None** ≡ maximum
        Number of GPU threads per blocks.
    """

    ####################################################################################################################

    M_MIN = 0.5
    M_MAX = 1.5
    M_NB_OF_STEPS = 8

    ####################################################################################################################

    Σ_MIN = 1.0
    Σ_MAX = 8.0
    Σ_NB_OF_STEPS = 8

    ####################################################################################################################

    ALPHA_LIST = [0.05, 0.1, 0.2, 0.3, 0.4]

    ####################################################################################################################

    def __init__(self, dataset: typing.Union[np.ndarray, typing.Callable], dataset_weights: typing.Optional[typing.Union[np.ndarray, typing.Callable]] = None, topology: typing.Optional[str] = None, batch: bool = True, use_best_epoch: bool = True, show_progress_bar: bool = True, enable_gpu: bool = True, threads_per_blocks: typing.Optional[int] = None):

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

        """
        Finds the best hyperparameters.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to train for.

        Returns
        -------
        int
            Suggested number of neuron rows and columns.
        float
            Suggested learning rate (if batch is **False**).
        float
            Suggested neighborhood radius.
        int
            Index of the selected quantization error.
        np.ndarray
            Array of quantization errors.
        np.ndarray
            Array of topographic errors.
        """

        ################################################################################################################

        result_m = 0
        result_α = math.nan
        result_σ = math.nan
        result_idx = -1
        result_qes = np.empty(shape = 0, dtype = np.float64)
        result_tes = np.empty(shape = 0, dtype = np.float64)

        min_qe = math.inf

        ################################################################################################################
        # COMPUTE M_REF                                                                                                #
        ################################################################################################################

        size = 0

        if self._dataset_weights is None:

            dataset_generator_builder = dataset_to_generator_builder(self._dataset)

            dataset_generator = dataset_generator_builder()

            for vectors in dataset_generator():

                size += vectors.shape[0]

        else:

            dataset_weight_generator_builder = dataset_to_generator_builder(self._dataset_weights)

            dataset_weight_generator = dataset_weight_generator_builder()

            for dataset_weights in dataset_weight_generator():

                size += np.sum(dataset_weights)

        ################################################################################################################

        m_ref = np.sqrt(5.0 * np.sqrt(size))

        ################################################################################################################
        # SCAN HYPERPARAMETERS                                                                                         #
        ################################################################################################################

        with tqdm.tqdm(total = HypParamFinder_SOM.M_NB_OF_STEPS * HypParamFinder_SOM.Σ_NB_OF_STEPS * (1 if self._batch else len(HypParamFinder_SOM.ALPHA_LIST)), disable = not self._show_progress_bar) as pbar:

            m_list = np.unique(np.linspace(HypParamFinder_SOM.M_MAX * m_ref, HypParamFinder_SOM.M_MIN * m_ref, num = HypParamFinder_SOM.M_NB_OF_STEPS, dtype = float))
            for m in m_list:

                σ_list = np.unique(m / np.linspace(HypParamFinder_SOM.Σ_MAX, HypParamFinder_SOM.Σ_MIN, num = HypParamFinder_SOM.Σ_NB_OF_STEPS, dtype = float))
                for σ in σ_list:

                    for α in ([None] if self._batch else HypParamFinder_SOM.ALPHA_LIST):

                        m = round(m)
                        σ = float(σ)

                        idx, qes, tes = self._train_step2(m, α,  σ, n_epochs)

                        qe = qes[idx]

                        if min_qe > qe:

                            min_qe = qe

                            result_m = m
                            result_α = α
                            result_σ = σ
                            result_idx = idx
                            result_qes = qes
                            result_tes = tes

                        pbar.update(1)

        ################################################################################################################

        return result_m, result_α, result_σ, result_idx, result_qes, result_tes

    ####################################################################################################################

    def _train_step2(self, m: int, alpha: typing.Optional[float], sigma: float, n_epochs: int):

        ################################################################################################################
        # PCA                                                                                                          #
        ################################################################################################################

        pca = som_pca.SOM_PCA(m, m, self._dataset.shape[1], dtype = self._dataset.dtype, topology = self._topology)

        ################################################################################################################

        pca.train(self._dataset, dataset_weights = self._dataset_weights, min_weight = 0.0, max_weight = 1.0, show_progress_bar = False)

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
            som.train(self._dataset, dataset_weights = self._dataset_weights, n_epochs = n_epochs, n_vectors = None, use_best_epoch = self._use_best_epoch, show_progress_bar = False, enable_gpu = self._enable_gpu, threads_per_blocks = self._threads_per_blocks)
        else:
            som.train(self._dataset, dataset_weights = self._dataset_weights, n_epochs = n_epochs, n_vectors = None, use_best_epoch = self._use_best_epoch, show_progress_bar = False)

        ################################################################################################################
        # GET QE                                                                                                       #
        ################################################################################################################

        return np.argmin(som.quantization_errors) if self._use_best_epoch else som.quantization_errors.shape[0] - 1, som.quantization_errors, som.topographic_errors

########################################################################################################################
