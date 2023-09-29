#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import numpy as np

import decontamination

import matplotlib.pyplot as plt

########################################################################################################################

for topology in ['square', 'hexagonal']:

    weights = np.array([[0, 1], [2, 3], [4, 5]])

    fig, ax = decontamination.display_latent_space(weights, log_scale = False, topology = topology)

    ax.set_xlabel('n, j, y')
    ax.set_ylabel('m, i, x')

    weights = np.array([[0, 1], [1, 10], [100, 1000]])

    fig, ax = decontamination.display_latent_space(weights, log_scale = True, topology = topology)

    ax.set_xlabel('n, j, y')
    ax.set_ylabel('m, i, x')

########################################################################################################################

if __name__ == '__main__':

    plt.tight_layout()

    plt.show()

########################################################################################################################
