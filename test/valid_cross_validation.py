#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

from sklearn.linear_model import ElasticNet as SklearnElasticNet

from decontamination.algo import batch_iterator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import decontamination

import numpy as np

########################################################################################################################

true_coef = np.array([1.5, -2.0, 3.0, 0, 0, 0, 0.1, 0, 0, 0], dtype = np.float64)

########################################################################################################################

rnd = np.random.default_rng(seed = 0)

X_train = rnd.standard_normal((160, 10), dtype = np.float64)
Y_train = np.dot(X_train, true_coef) + rnd.standard_normal(160, dtype = np.float64) * 0.5

X_test = rnd.standard_normal((40, 10), dtype = np.float64)
Y_test = np.dot(X_test, true_coef) + rnd.standard_normal(40, dtype = np.float64) * 0.5

########################################################################################################################

def generator_builder():

    def generator():

        for s, e in batch_iterator(X_train.shape[0], 16):

            yield (
                X_train[s: e],
                Y_train[s: e],
            )

    return generator

########################################################################################################################

def test_cross_validation():

    model_enetcv = decontamination.CrossValidation_ElasticNet(dim = 10, max_batch_iter = 100, l1_ratios = [0.1, 0.5, 0.9], n_rhos = 20, eps = 1e-4, cv = 5)

    result = model_enetcv.find_hyper_parameters(generator_builder, n_epochs = 100, soft_thresholding = True, show_progress_bar = True)

    print(result)

########################################################################################################################

if __name__ == '__main__':

    test_cross_validation()

########################################################################################################################
