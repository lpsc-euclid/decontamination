#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

from sklearn.linear_model import ElasticNetCV as SklearnElasticNetCV

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

    batched = True

    print('* DECONTAMINATION *')

    cv = decontamination.CrossValidation_ElasticNet(dim = 10, l1_ratios = [0.1, 0.5, 0.9], n_rhos = 20, eps = 1e-4, cv = 5, alpha = 0.005, tolerance = 1.0e-4)

    result = cv.find_hyper_parameters(generator_builder if batched else (X_train, Y_train), n_epochs = 1000, soft_thresholding = True, show_progress_bar = True)

    print(result)

    print('* SKLEARN *')

    reg = SklearnElasticNetCV(l1_ratio = [0.1, 0.5, 0.9], n_alphas = 20, eps = 1e-4, cv = 5, tol = 1.0e-4, max_iter = 1000)

    reg.fit(X = X_train, y = Y_train)

    print(reg.alpha_, reg.l1_ratio_)

########################################################################################################################

if __name__ == '__main__':

    test_cross_validation()

########################################################################################################################
