#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import unittest
import decontamination

import numpy as np

########################################################################################################################

class JITTests(unittest.TestCase):

    ####################################################################################################################

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        ##

        np.random.seed(0)

        ##

        self.som = decontamination.SOM_PCA(4, 4, 4, np.float32, topology = 'square')

        self.data = np.random.randn(100_000).reshape(25_000, 4)

        self.som.train(self.data)

    ####################################################################################################################

    def test1(self):

        expected = np.array([
            [[-1.08487280, -0.32087362, +0.75769144, +0.38209260],
             [-0.87059164, -0.42448060, +0.13506313, +0.39338970],
             [-0.65631050, -0.52808756, -0.48756516, +0.40468680],
             [-0.44202930, -0.63169456, -1.11019350, +0.41598392]],

            [[-0.57590544, -0.00335089, +0.87519210, +0.11606709],
             [-0.36162427, -0.10695787, +0.25256380, +0.12736419],
             [-0.14734310, -0.21056485, -0.37006450, +0.13866130],
             [+0.06693809, -0.31417182, -0.99269277, +0.14995840]],

            [[-0.06693809, +0.31417182, +0.99269277, -0.14995840],
             [+0.14734310, +0.21056485, +0.37006450, -0.13866130],
             [+0.36162427, +0.10695787, -0.25256380, -0.12736419],
             [+0.57590544, +0.00335089, -0.87519210, -0.11606709]],

            [[+0.44202930, +0.63169456, +1.11019350, -0.41598392],
             [+0.65631050, +0.52808756, +0.48756516, -0.40468680],
             [+0.87059164, +0.42448060, -0.13506313, -0.39338970],
             [+1.08487280, +0.32087362, -0.75769144, -0.38209260]]
        ])

        self.assertTrue(np.allclose(self.som.get_centroids(), expected))

########################################################################################################################

if __name__ == '__main__':

    unittest.main()

########################################################################################################################
