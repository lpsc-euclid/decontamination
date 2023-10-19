# -*- coding: utf-8 -*-
########################################################################################################################

import typing

import numpy as np

from ..algo import som_pca, som_batch, som_online, clustering

########################################################################################################################

# noinspection PyPep8Naming
class Decontamination_SOM(object):

    """
    ???

    Parameters
    ----------
    m : int
        Number of neuron rows.
    n : int
        Number of neuron columns.
    dim : int
        Dimensionality of the input data.
    batch : bool
        Specifies whether to train parallel (**True**) or iterative (**False**).
    dtype : typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]
        Neural network data type, either **np.float32** or **np.float64** (default: **np.float32**).
    topology : typing.Optional[str]
        Topology of the model, either **'square'** or **'hexagonal'** (default: **None** ≡ **'hexagonal'**).
    alpha : float
        Starting value of the learning rate (default: **None** ≡ **0.3**, iterative training only).
    sigma : float
        Starting value of the neighborhood radius (default: **None** ≡ :math:`\\mathrm{max}(m,n)/2`).
    """

    ####################################################################################################################

    def __init__(self, m: int, n: int, dim: int, batch: bool, dtype: typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]] = np.float32, topology: typing.Optional[str] = 'hexagonal', alpha: float = None, sigma: float = None):

        ################################################################################################################
        # PCA                                                                                                          #
        ################################################################################################################

        self._pca = som_pca.SOM_PCA(m, n, dim, dtype = dtype, topology = topology)

        ################################################################################################################
        # SOM                                                                                                          #
        ################################################################################################################

        self._batch = batch
        self._catalog_systematics = None
        self._footprint_systematics = None

        if batch:
            self._som = som_batch.SOM_Batch(m, n, dim, dtype = dtype, topology = topology, sigma = sigma)
        else:
            self._som = som_online.SOM_Online(m, n, dim, dtype = dtype, topology = topology, alpha = alpha, sigma = sigma)

        ################################################################################################################
        # OTHER                                                                                                        #
        ################################################################################################################

        self._winners = None

        self._cluster_ids = None

        ################################################################################################################

        self._catalog_activation_map = None
        self._clustered_catalog_activation_map = None

        self._footprint_activation_map = None
        self._clustered_footprint_activation_map = None

        ################################################################################################################

        self._n_gal = None
        self._n_pix = None

        ################################################################################################################

        self._gnd = None
        self._clustered_gnd = None

        self._gndm = None
        self._clustered_gndm = None

    ####################################################################################################################

    @property
    def m(self) -> int:

        """Number of neuron rows."""

        return self._som.m

    ####################################################################################################################

    @property
    def n(self) -> int:

        """Number of neuron columns."""

        return self._som.n

    ####################################################################################################################

    @property
    def dim(self) -> int:

        """Dimensionality of the input data."""

        return self._som.dim

    ####################################################################################################################

    @property
    def dtype(self) -> typing.Type[typing.Union[np.float32, np.float64, float, np.int32, np.int64, int]]:

        """Neural network data type."""

        return self._som.dtype

    ####################################################################################################################

    @property
    def topology(self) -> str:

        """Model topology, either **'square'** or **'hexagonal'**."""

        return self._som.topology

    ####################################################################################################################

    @property
    def alpha(self) -> str:

        """Starting value of the learning rate."""

        return self._som.alpha

    ####################################################################################################################

    @property
    def sigma(self) -> str:

        """Starting value of the neighborhood radius."""

        return self._som.sigma

    ####################################################################################################################

    @property
    def quantization_errors(self) -> np.ndarray:

        """Quantization errors."""

        return self._som.get_quantization_errors()

    ####################################################################################################################

    @property
    def topographic_errors(self) -> np.ndarray:

        """Topographic errors."""

        return self._som.get_topographic_errors()

    ####################################################################################################################

    @property
    def weights(self) -> np.ndarray:

        """Weights in the latent space with the shape `[m * n, dim]`."""

        return self._som.get_weights()

    ####################################################################################################################

    @property
    def centroids(self) -> np.ndarray:

        """Weights in the latent space with the shape `[m, n, dim]`."""

        return self._som.get_centroids()

    ####################################################################################################################

    @property
    def winners(self) -> np.ndarray:

        """Winners for footprint."""

        return self._winners

    ####################################################################################################################

    @property
    def cluster_ids(self) -> np.ndarray:

        """Cluster identifiers for neurons."""

        return self._cluster_ids

    ####################################################################################################################

    @property
    def catalog_activation_map(self) -> np.ndarray:

        """Activation map for catalog."""

        return self._catalog_activation_map

    ####################################################################################################################

    @property
    def clustered_catalog_activation_map(self) -> np.ndarray:

        """Clustered activation map for catalog."""

        return self._clustered_catalog_activation_map

    ####################################################################################################################

    @property
    def footprint_activation_map(self) -> np.ndarray:

        """Activation map for footprint."""

        return self._footprint_activation_map

    ####################################################################################################################

    @property
    def clustered_footprint_activation_map(self) -> np.ndarray:

        """Clustered activation map for footprint."""

        return self._clustered_footprint_activation_map

    ####################################################################################################################

    @property
    def n_gal(self) -> np.ndarray:

        """Number of galaxies."""

        return self._n_gal

    ####################################################################################################################

    @property
    def n_pix(self) -> np.ndarray:

        """Number of pixels."""

        return self._n_pix

    ####################################################################################################################

    @property
    def gnd(self) -> np.ndarray:

        """
        Galaxy Number Density (GND) in the latent space.

        .. math::
            \\mathrm{gnd}\\equiv\\frac{\\mathrm{catalog\\ activation\\ map}}{\\mathrm{footprint\\ activation\\ map}}
        """

        return self._gnd

    ####################################################################################################################

    @property
    def clustered_gnd(self) -> np.ndarray:

        """
        Clustered Galaxy Number Density (GND) in the latent space.

        .. math::
            \\mathrm{clustered\\ gnd}\\equiv\\frac{\\mathrm{clustered\\ catalog\\ activation\\ map}}{\\mathrm{clustered\\ footprint\\ activation\\ map}}
        """

        return self._clustered_gnd

    ####################################################################################################################

    @property
    def gndc(self) -> np.ndarray:

        """
        Galaxy Number Density Contrast (GNDC) in the latent space.

        .. math::
            \\mathrm{gndc}\\equiv\\frac{\\mathrm{gnd}-\\frac{n_\\mathrm{gal}}{n_\\mathrm{pix}}}{\\frac{n_\\mathrm{gal}}{n_\\mathrm{pix}}}
        """

        return (self._gnd - (self._n_gal / self._n_pix)) / (self._n_gal / self._n_pix)

    ####################################################################################################################

    @property
    def clustered_gndc(self) -> np.ndarray:

        """
        Clustered Galaxy Number Density Contrast (GNDC) in the latent space.

        .. math::
            \\mathrm{clustered\\ gndc}\\equiv\\frac{\\mathrm{clustered\\ gnd}-\\frac{n_\\mathrm{gal}}{n_\\mathrm{pix}}}{\\frac{n_\\mathrm{gal}}{n_\\mathrm{pix}}}
        """

        return (self._clustered_gnd - (self._n_gal / self._n_pix)) / (self._n_gal / self._n_pix)

    ####################################################################################################################

    @property
    def gndm(self) -> np.ndarray:

        """Galaxy Number Density Map (GNDM) in the physical space."""

        return self._gndm

    ####################################################################################################################

    @property
    def clustered_gndm(self) -> np.ndarray:

        """Clustered Galaxy Number Density Map (GNDM) in the physical space."""

        return self._clustered_gndm

    ####################################################################################################################

    @property
    def gndcm(self) -> np.ndarray:

        """
        Galaxy Number Density Contrast Map (GNDCM) in the physical space.

        .. math::
            \\mathrm{gndcm}\\equiv\\frac{\\mathrm{gndm}-\\frac{n_\\mathrm{gal}}{n_\\mathrm{pix}}}{\\frac{n_\\mathrm{gal}}{n_\\mathrm{pix}}}
        """

        return (self._gndm - (self._n_gal / self._n_pix)) / (self._n_gal / self._n_pix)

    ####################################################################################################################

    @property
    def clustered_gndcm(self) -> np.ndarray:

        """
        Clustered Galaxy Number Density Contrast Map (GNDCM) in the physical space.

        .. math::
            \\mathrm{clustered\\ gndcm}\\equiv\\frac{\\mathrm{clustered\\ gndm}-\\frac{n_\\mathrm{gal}}{n_\\mathrm{pix}}}{\\frac{n_\\mathrm{gal}}{n_\\mathrm{pix}}}
        """

        return (self._clustered_gndm - (self._n_gal / self._n_pix)) / (self._n_gal / self._n_pix)

    ####################################################################################################################

    # noinspection PyArgumentList
    def train(self, catalog_systematics: typing.Union[np.ndarray, typing.Callable], footprint_systematics: typing.Union[np.ndarray, typing.Callable], n_epochs: typing.Optional[int] = None, n_vectors: typing.Optional[int] = None, n_error_bins: int = 10, show_progress_bar: bool = True, enable_gpu: bool = True, threads_per_blocks: int = 1024) -> None:

        """
        ???

        Parameters
        ----------
        catalog_systematics : typing.Union[np.ndarray, typing.Callable]
            Dataset array or generator builder of systematics for the catalog.
        footprint_systematics : typing.Union[np.ndarray, typing.Callable]
            Dataset array or generator builder of systematics for the footprint.
        n_epochs : typing.Optional[int]
            Number of epochs to train for (default: **None**).
        n_vectors : typing.Optional[int]
            Number of vectors to train for (default: **None**).
        n_error_bins : int
            Number of quantization and topographic error bins (default: **10**).
        show_progress_bar : bool
            Specifies whether to display a progress bar (default: **True**).
        enable_gpu : bool
            If available, run on GPU rather than CPU (default: **True**).
        threads_per_blocks : int
            Number of GPU threads per blocks (default: **1024**).
        """

        ################################################################################################################
        # SET DATASETS                                                                                                 #
        ################################################################################################################

        self._catalog_systematics = catalog_systematics
        self._footprint_systematics = footprint_systematics

        ################################################################################################################
        # PCA TRAINING                                                                                                 #
        ################################################################################################################

        self._pca.train(catalog_systematics, min_weight = 0.0, max_weight = 1.0)

        ################################################################################################################
        # BATCH/ONLINE TRAINING                                                                                        #
        ################################################################################################################

        self._som.init_from(self._pca)

        if self._batch:
            self._som.train(catalog_systematics, n_epochs = n_epochs, n_vectors = n_vectors, n_error_bins = n_error_bins, show_progress_bar = show_progress_bar, enable_gpu = enable_gpu, threads_per_blocks = threads_per_blocks)
        else:
            self._som.train(catalog_systematics, n_epochs = n_epochs, n_vectors = n_vectors, n_error_bins = n_error_bins, show_progress_bar = show_progress_bar)

        ################################################################################################################
        # COMPUTE FOOTPRINT WINNERS                                                                                    #
        ################################################################################################################

        self._winners = self._som.get_winners(footprint_systematics, enable_gpu = enable_gpu, threads_per_blocks = threads_per_blocks)

    ####################################################################################################################

    def _process(self, catalog_activation_map: np.ndarray, footprint_activation_map: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

        ################################################################################################################
        # COMPUTE NUMBER OF GALAXIES & PIXELS                                                                          #
        ################################################################################################################

        self._n_gal = np.sum(catalog_activation_map)
        self._n_pix = np.sum(footprint_activation_map)

        ################################################################################################################
        # COMPUTE GALAXY NUMBER DENSITY                                                                                #
        ################################################################################################################

        gnd = np.divide(
            catalog_activation_map,
            footprint_activation_map,
            out = np.full(catalog_activation_map.shape, np.nan, dtype = self._som.dtype),
            where = footprint_activation_map > 0.0
        )

        ################################################################################################################
        # COMPUTE GALAXY NUMBER DENSITY MAP                                                                            #
        ################################################################################################################

        gndm = gnd.reshape(self._som.m * self._som.n)[self._winners]

        ################################################################################################################

        return gnd, gndm

    ####################################################################################################################

    def process_flat(self, enable_gpu: bool = True, threads_per_blocks: int = 1024) -> None:

        """
        ???

        Parameters
        ----------
        enable_gpu : bool
            If available, run on GPU rather than CPU (default: **True**).
        threads_per_blocks : int
            Number of GPU threads per blocks (default: **1024**).
        """

        if self._catalog_systematics is None\
           or                                \
           self._footprint_systematics is None:

            raise Exception('Underlying SOM network not trained')

        ################################################################################################################
        # COMPUTE ACTIVATION MAPS                                                                                      #
        ################################################################################################################

        self._catalog_activation_map = self._som.get_activation_map(self._catalog_systematics, enable_gpu = enable_gpu, threads_per_blocks = threads_per_blocks)
        self._footprint_activation_map = self._som.get_activation_map(self._footprint_systematics, enable_gpu = enable_gpu, threads_per_blocks = threads_per_blocks)

        ################################################################################################################
        # COMPUTE GALAXY NUMBER DENSITY XXX                                                                            #
        ################################################################################################################

        self._gnd, self._gndm = self._process(
            self._catalog_activation_map,
            self._footprint_activation_map
        )

    ####################################################################################################################

    def process_clustered(self, n_clusters: int, enable_gpu: bool = True, threads_per_blocks: int = 1024) -> None:

        """
        ???

        Parameters
        ----------
        n_clusters : int
            Desired number latent space clusters.
        enable_gpu : bool
            If available, run on GPU rather than CPU (default: **True**).
        threads_per_blocks : int
            Number of GPU threads per blocks (default: **1024**).
        """

        m = self._som.m
        n = self._som.n

        if self._catalog_activation_map is None\
           or                                   \
           self._footprint_activation_map is None:

            self.process_flat(enable_gpu = enable_gpu, threads_per_blocks = threads_per_blocks)

        ################################################################################################################
        # CLUSTER LATENT SPACE                                                                                         #
        ################################################################################################################

        weights = self._som.get_weights().copy()

        weights[self._footprint_activation_map.reshape(m * n) == 0] = np.nan

        self._cluster_ids = clustering.Clustering.clusterize(weights, n_clusters)

        ################################################################################################################
        # COMPUTE CLUSTERED ACTIVATION MAPS                                                                            #
        ################################################################################################################

        self._clustered_catalog_activation_map = clustering.Clustering.average(self._catalog_activation_map.reshape(m * n), self._cluster_ids).reshape(m, n)
        self._clustered_footprint_activation_map = clustering.Clustering.average(self._footprint_activation_map.reshape(m * n), self._cluster_ids).reshape(m, n)

        ################################################################################################################
        # COMPUTE CLUSTERED GALAXY NUMBER DENSITY                                                                      #
        ################################################################################################################

        self._clustered_gnd, self._clustered_gndm = self._process(
            self._clustered_catalog_activation_map,
            self._clustered_footprint_activation_map
        )

########################################################################################################################
