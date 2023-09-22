########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import minisom
import decontamination

import numpy as np

import matplotlib.pyplot as plt

########################################################################################################################

M = 30
N = 30
SEED = 10
DATASET_SIZE = 25_000

########################################################################################################################

data = np.random.default_rng(seed = SEED).random((DATASET_SIZE, 4), dtype = np.float32)

########################################################################################################################

som_ref = minisom.MiniSom(M, N, 4, learning_rate = 0.3, sigma = max(M, N) / 2)

som_ref.pca_weights_init(data)

########################################################################################################################

som_new = decontamination.SOM_PCA(M, N, 4)

som_new.train(data)

########################################################################################################################

fig, axs = plt.subplots(4, 4, figsize = (8, 8))

########################################################################################################################

for i in range(2):
    for j in range(2):

        dimension = i * 2 + j

        weights_ref = som_ref.get_weights()[:, :, dimension]
        weights_new = som_new.get_centroids()[:, :, dimension]

        weights_diff = (weights_ref - weights_new) / weights_ref

        max_ref = np.max(np.abs(weights_ref))
        max_diff = np.max(np.abs(weights_diff))

        img = axs[i + 0, j + 0].imshow(weights_ref, cmap = 'viridis', vmin = -max_ref, vmax = +max_ref)
        fig.colorbar(img, ax = axs[i + 0, j + 0])
        axs[i + 0, j + 0].set_ylabel(f'w_{dimension + 1} PCA')

        img = axs[i + 2, j + 0].imshow(weights_diff, cmap = 'viridis', vmin = -max_diff, vmax = +max_diff)
        fig.colorbar(img, ax = axs[i + 2, j + 0])
        axs[i + 2, j + 0].set_ylabel(f'ref/new w_{dimension + 1} PCA')

########################################################################################################################

som_ref.train(data, data.shape[0])

########################################################################################################################

som_next = decontamination.SOM_Online(M, N, 4, alpha = 0.3, sigma = max(M, N) / 2.0)

som_next.init_from(som_new)

som_next.train(data)

########################################################################################################################

for i in range(2):
    for j in range(2):

        dimension = i * 2 + j

        weights_ref = som_ref.get_weights()[:, :, dimension]
        weights_new = som_next.get_centroids()[:, :, dimension]

        weights_diff = (weights_ref - weights_new) / weights_ref

        max_ref = np.max(np.abs(weights_ref))
        max_diff = np.max(np.abs(weights_diff))

        img = axs[i + 0, j + 2].imshow(weights_ref, cmap = 'viridis', vmin = -max_ref, vmax = +max_ref)
        fig.colorbar(img, ax = axs[i + 0, j + 2])
        axs[i + 0, j + 2].set_ylabel(f'w_{dimension + 1} online')

        img = axs[i + 2, j + 2].imshow(weights_diff, cmap = 'viridis', vmin = -max_diff, vmax = +max_diff)
        fig.colorbar(img, ax = axs[i + 2, j + 2])
        axs[i + 2, j + 2].set_ylabel(f'ref/new w_{dimension + 1} online')

########################################################################################################################

plt.tight_layout()

plt.show()

########################################################################################################################
