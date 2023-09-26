# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from . import som_abstract

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
        A rule of thumb to set the size of the grid for a dimensionality reduction
        task is that it should contain \\( 5\\sqrt{N} \\) neurons where N is the
        number of samples in the dataset to analyze.

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
            Topology of the model, either **'square'** or **'hexagonal'** (default: **'hexagonal'**).
        """

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology)

        ################################################################################################################

        self._n_epochs = None

        self._n_vectors = None

        ################################################################################################################

        self._header_extra = {
            'mode': '__MODE__',
            'n_epochs': '_n_epochs',
            'n_vectors': '_n_vectors',
        }

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], n_epochs: typing.Optional[int] = None, n_vectors: typing.Optional[int] = None, n_error_bins: typing.Optional[int] = 10, show_progress_bar: bool = False) -> None:

        """
        Trains the neural network. Use either the "*number of epochs*" training method by specifying `n_epochs` (then \\( e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_epochs}\\}-1 \\)) or the "*number of vectors*" training method by specifying `n_vectors` (then \\( e\\equiv 0\\dots\\{e_\\mathrm{tot}\\equiv\\mathrm{n\\_vectors}\\}-1 \\)). A batch formulation of updating weights is implemented: $$ c_i(e)\\equiv\\mathrm{bmu}(x_i,e)\\equiv\\underset{j}{\\mathrm{arg\\,min}}\\lVert x_i-w_j(e)\\rVert $$ $$ n_{ji}(e)=\\left\\{\\begin{array}{ll}1&j=c_i(e)\\\\0&\\mathrm{otherwise}\\end{array}\\right. $$ $$ \\boxed{w_j(e+1)=\\frac{\\sum_{i=0}^{N-1}n_{ji}(e)x_i}{\\sum_{i=0}^{N-1}n_{ji}(e)}} $$ where \\( j=0\\dots m\\times n-1 \\).

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array or generator builder.
        n_epochs : typing.Optional[int]
            Number of epochs to train for (default: **None**).
        n_vectors : typing.Optional[int]
            Number of vectors to train for (default: **None**).
        n_error_bins : int
            Number of error bins (default: **10**).
        show_progress_bar : bool
            Specifies whether to display a progress bar (default: **False**).
        """

        ################################################################################################################

        pass

########################################################################################################################
