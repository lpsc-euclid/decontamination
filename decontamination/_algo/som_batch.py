# -*- coding: utf-8 -*-
########################################################################################################################

import tqdm
import typing

import numpy as np

from .. import jit, device_array_empty, device_array_zeros

from . import som_abstract, dataset_to_generator_builder

########################################################################################################################

# noinspection PyPep8Naming
class SOM_Batch(som_abstract.SOM_Abstract):

    """
    Self Organizing Maps (standard batch implementation).
    """

    __MODE__ = 'batch'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[np.single] = np.float32, topology: typing.Optional[str] = None):

        """
        A rule of thumb to set the size of the grid for a dimensionality reduction task is that it should contain \\( 5\\sqrt{N} \\) neurons where N is the number of samples in the dataset to analyze.

        Parameters
        ----------
        m : int
            Number of neuron rows.
        n : int
            Number of neuron columns.
        dim : int
            Dimensionality of the input data.
        dtype : typing.Type[np.single]
            Neural network data type (default: **np.float32**).
        topology : typing.Optional[str]
            Topology of the model, either **'square'** or **'hexagonal'** (default: **None**, uses: **'hexagonal'**).
        """

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology)

        ################################################################################################################

        self._n_epochs = None

        ################################################################################################################

        self._header_extra = {
            'mode': '__MODE__',
            'n_epochs': '_n_epochs',
        }

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], n_epochs: int, show_progress_bar: bool = False, enable_gpu: bool = True, threads_per_blocks: int = 1024) -> None:

        """
        Trains the neural network. Use either the "*number of epochs*" training method by specifying `n_epochs` (then \\( e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_epochs}\\}-1 \\)) or the "*number of vectors*" training method by specifying `n_vectors` (then \\( e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_vectors}\\}-1 \\)). A batch formulation of updating weights is implemented: $$ c_i(w,e)\\equiv\\mathrm{bmu}(x_i,w,e)\\equiv\\underset{j}{\\mathrm{arg\\,min}}\\lVert x_i-w_j(e)\\rVert $$ $$ \\Theta_{ji}(w,e)\\equiv\\delta_{j,c_i(w,e)}\\equiv\\left\\{\\begin{array}{ll}1&j=c_i(w,e)\\\\0&\\mathrm{otherwise}\\end{array}\\right. $$ $$ \\boxed{w_j(e+1)=\\frac{\\sum_{i=0}^{N-1}\\Theta_{ji}(w,e)x_i}{\\sum_{i=0}^{N-1}\\Theta_{ji}(w,e)}} $$ where \\( j=0\\dots m\\times n-1 \\).

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        n_epochs : int
            Number of epochs to train for.
        show_progress_bar : bool
            Specifies whether to display a progress bar (default: **False**).
        enable_gpu : bool
            If available, run on GPU rather than CPU (default: **True**).
        threads_per_blocks : int
            Number of GPU threads per blocks (default: **1024**).
        """

        ################################################################################################################

        generator_builder = dataset_to_generator_builder(dataset)

        ################################################################################################################

        quantization_errors = device_array_empty(n_epochs, dtype = np.float32)

        topographic_errors = device_array_empty(n_epochs, dtype = np.float32)

        ################################################################################################################

        self._n_epochs = n_epochs

        ################################################################################################################

        for cur_epoch in tqdm.trange(n_epochs, disable = not show_progress_bar):

            ############################################################################################################

            generator = generator_builder()

            ############################################################################################################

            numerator = device_array_zeros(shape = (self._m * self._n,), dtype = self._dtype)

            denominator = device_array_zeros(shape = (self._m * self._n,), dtype = self._dtype)

            ############################################################################################################

            for data in generator():

                _train_kernel[enable_gpu, threads_per_blocks, data.shape[0]](
                    numerator,
                    denominator,
                    quantization_errors,
                    topographic_errors,
                    self._weights,
                    data,
                    cur_epoch
                )

            ############################################################################################################

            self._weights = numerator.copy_to_host() / denominator.copy_to_host()

        ################################################################################################################

        self._quantization_errors = quantization_errors.copy_to_host()

        self._topographic_errors = topographic_errors.copy_to_host()

########################################################################################################################

@jit(kernel = True, parallel = False)
def _train_kernel(numerator: np.ndarray, denominator: np.ndarray, quantization_errors: np.ndarray, topographic_errors: np.ndarray, weights: np.ndarray, vectors: np.ndarray, cur_epoch: int) -> None:

    pass

########################################################################################################################
