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
import numba as nb

from . import covariance, som_abstract

########################################################################################################################

covariance_diagonalize = covariance.Covariance.diagonalize

########################################################################################################################

# noinspection PyPep8Naming, DuplicatedCode
class SOM_PCA(som_abstract.SOM_Abstract):

    """
    Self Organizing Maps that span the first two principal components with constant memory usage.

    .. note::
        A rule of thumb to set the size of the grid for a dimensionality reduction task is that it should contain :math:`5\\sqrt{N}` neurons where N is the number of samples in the dataset to analyze.

    Parameters
    ----------
    m : int
        Number of neuron rows.
    n : int
        Number of neuron columns.
    dim : int
        Dimensionality of the input data.
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]], default: **np.float32**
        Neural network data type, either **np.float32** or **np.float64**.
    topology : str, default: **None** ≡ **'hexagonal'**
        Neural network topology, either **'square'** or **'hexagonal'**.
    """

    __MODE__ = 'pca'

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, topology: typing.Optional[str] = None):

        if dim < 2:

            raise ValueError('SOM_PCA requires dim >= 2.')

        ################################################################################################################

        super().__init__(m, n, dim, dtype, topology)

        ################################################################################################################

        self._scale_by_variance = False

        self._apply_cdf = False

        self._cdf_gain = 1.0

        ################################################################################################################

        self._cov_matrix = np.zeros((dim, dim), dtype = np.float64)

        self._eigenvalues = np.zeros((dim, ), dtype = np.float64)

        self._eigenvectors = np.zeros((dim, dim), dtype = np.float64)

        self._orders = np.zeros((dim, ), dtype = np.float64)

        ################################################################################################################

        self._header_extra = {
            'mode': '__MODE__',
            ##
            'scale_by_variance': '_scale_by_variance',
            'apply_cdf': '_apply_cdf',
            'cdf_gain': '_cdf_gain',
        }

    ####################################################################################################################

    @property
    def cov_matrix(self) -> np.ndarray:

        """Covariance matrix of the training dataset."""

        return self._cov_matrix

    ####################################################################################################################

    @property
    def eigenvalues(self) -> np.ndarray:

        """Eigenvalues of the covariance matrix."""

        return self._eigenvalues

    ####################################################################################################################

    @property
    def eigenvectors(self) -> np.ndarray:

        """Eigenvectors of the covariance matrix."""

        return self._eigenvectors

    ####################################################################################################################

    @property
    def orders(self) -> np.ndarray:

        """Order of importance of the components."""

        return self._orders

    ####################################################################################################################

    @staticmethod
    @nb.njit()
    def _diag_cov_matrix(cov_matrix: np.ndarray, min_weight: float, max_weight: float, m: int, n: int, dim: int, scale_by_variance: bool, apply_cdf: bool, cdf_gain: float) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        ################################################################################################################

        weights = np.empty((m, n, dim, ), dtype = np.float64)

        ################################################################################################################

        eigenvalues, eigenvectors, orders = covariance_diagonalize(cov_matrix, sort = True)

        ################################################################################################################

        v0 = eigenvectors[:, 0]
        v1 = eigenvectors[:, 1]

        if scale_by_variance:

            ev0 = eigenvalues[0]
            if ev0 > 0.0:
                v0 = v0 * np.sqrt(ev0)

            ev1 = eigenvalues[1]
            if ev1 > 0.0:
                v1 = v1 * np.sqrt(ev1)

        ################################################################################################################

        sigma = np.sqrt(v0 * v0 + v1 * v1)

        sigma[sigma == 0.0] = 1.0

        cdf_scale = cdf_gain / (math.sqrt(2.0) * sigma)

        ################################################################################################################

        if apply_cdf:

            linspace_x = np.linspace(-1.0, +1.0, m)
            linspace_y = np.linspace(-1.0, +1.0, n)

        else:

            linspace_x = np.linspace(min_weight, max_weight, m)
            linspace_y = np.linspace(min_weight, max_weight, n)

        ################################################################################################################

        for i in range(m):
            c1 = linspace_x[i]

            for j in range(n):
                c2 = linspace_y[j]

                for d in range(dim):

                    w = v0[d] * c1 + v1[d] * c2

                    weights[i, j, d] = min_weight + (max_weight - min_weight) * 0.5 * (1.0 + math.erf(w * cdf_scale[d])) if apply_cdf else w

        ################################################################################################################

        return eigenvalues, eigenvectors, orders, weights

    ####################################################################################################################

    def train(self, dataset: typing.Union[np.ndarray, typing.Callable], dataset_weights: typing.Optional[typing.Union[np.ndarray, typing.Callable]] = None, min_weight: float = 0.0, max_weight: float = 1.0, scale_by_variance: bool = False, apply_cdf: bool = False, cdf_gain: float = 1.0, show_progress_bar: bool = False) -> None:

        """
        Trains the neural network from the given dataset.

        Parameters
        ----------
        dataset : typing.Union[np.ndarray, typing.Callable]
            Training dataset array of shape :math:`(N_\\mathrm{samples},\\mathrm{dim})` or generator builder.
        dataset_weights : typing.Union[np.ndarray, typing.Callable], default: **None**
            Training dataset weight array of shape :math:`(N_\\mathrm{samples},)` or generator builder.
        min_weight : float, default: **0.0**
            Latent space minimum value.
        max_weight : float, default: **1.0**
            Latent space maximum value.
        scale_by_variance : bool, default: **False**
            If **True**, scales the two principal directions by :math:`\\sigma_{k=\\{1,2\\}}=\\sqrt{\\lambda_k}` (before the optional CDF mapping).
        apply_cdf : bool, default: **False**
            If **True**, applies a per-component Gaussian CDF and rescales to :math:`]\\mathrm{min\\_weight},\\mathrm{max\\_weight}[`.
        cdf_gain : float, default: **1.0**
            When **apply_cdf** is **True**, gain applied in Gaussian CDF. Larger values increase saturation.
                .. math::
                    w_{i,j,k}=\\mathrm{min\\_weight}+(\\mathrm{max\\_weight}-\\mathrm{min\\_weight})\\frac{1}{2}\\left[1+\\mathrm{erf}\\left(\\frac{\\mathrm{cdf\\_gain}\\times w^\\mathrm{orig}_{i,j,k}}{\\sqrt{2}\\sigma_k}\\right)\\right]
        show_progress_bar : bool, default: **False**
            Specifies whether to display a progress bar.
        """

        ################################################################################################################

        cov_matrix = covariance.Covariance.compute(self._dim, dataset, dataset_weights, show_progress_bar = show_progress_bar)

        ################################################################################################################

        self.train_from_cov_matrix(
            cov_matrix,
            min_weight = min_weight,
            max_weight = max_weight,
            scale_by_variance = scale_by_variance,
            apply_cdf = apply_cdf,
            cdf_gain = cdf_gain,
            show_progress_bar = show_progress_bar
        )

    ####################################################################################################################

    def train_from_cov_matrix(self, cov_matrix: np.ndarray, min_weight: float = 0.0, max_weight: float = 1.0, scale_by_variance: bool = False, apply_cdf: bool = False, cdf_gain: float = 1.0, show_progress_bar: bool = False) -> None:

        """
        Trains the neural network from the given covariance matrix precomputed using class :class:`Covariance <decontamination.algo.covariance.Covariance>`

        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix.
        min_weight : float, default: **0.0**
            Latent space minimum value.
        max_weight : float, default: **1.0**
            Latent space maximum value.
        scale_by_variance : bool
            If **True**, scales the two principal directions by :math:`\\sqrt{\\lambda_i}` (before the optional CDF mapping).
        apply_cdf : bool, default: **False**
            If **True**, applies a per-component Gaussian CDF and rescales to :math:`[\\mathrm{min\\_weight},\\mathrm{max\\_weight}]`.
        cdf_gain : float, default: **1.0**
            When **apply_cdf** is **True**, gain applied in :math:`\\Phi(\\mathrm{cdf\\_gain}\\times x/\\sigma)`. Larger values increase saturation.
        show_progress_bar : bool, default: **False**
            Specifies whether to display a progress bar.
        """

        if max_weight <= min_weight:

            raise ValueError('max_weight must be > min_weight.')

        if apply_cdf and cdf_gain <= 0.0:

            raise ValueError('cdf_gain must be > 0 when apply_cdf is True.')

        ################################################################################################################

        self._cov_matrix[:] = cov_matrix

        ################################################################################################################

        eigenvalues64, eigenvectors64, orders64, weights64 = SOM_PCA._diag_cov_matrix(
            self._cov_matrix,
            min_weight,
            max_weight,
            self._m,
            self._n,
            self._dim,
            scale_by_variance = scale_by_variance,
            apply_cdf = apply_cdf,
            cdf_gain = cdf_gain
        )

        ################################################################################################################

        self._eigenvalues[:] = eigenvalues64
        self._eigenvectors[:] = eigenvectors64

        self._orders[:] = orders64

        ################################################################################################################

        self._weights[:] = weights64.astype(self._dtype, copy = False).reshape(self._weights.shape)

########################################################################################################################
