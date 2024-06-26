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

import matplotlib.pyplot as plt

########################################################################################################################

true_coef = np.array([1.5, -2.0, 3.0, 0, 0, 0, 0.1, 0, 0, 0], dtype = np.float32)

########################################################################################################################

rnd = np.random.default_rng(seed = 0)

X_train = rnd.standard_normal((160, 10), dtype = np.float32)
Y_train = np.dot(X_train, true_coef) + rnd.standard_normal(160, dtype = np.float32) * 0.5

X_test = rnd.standard_normal((40, 10), dtype = np.float32)
Y_test = np.dot(X_test, true_coef) + rnd.standard_normal(40, dtype = np.float32) * 0.5

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

def do_regressions(use_generator_builder):

    iter = 1000

    rho = 0.02
    l1_ratio = 0.9

    print('USE GENERATOR BUILDER', use_generator_builder)

    model_basic = decontamination.Regression_Basic(10, dtype = np.float32, alpha = 0.01, tolerance = None)
    model_basic.train(generator_builder if use_generator_builder else (X_train, Y_train), n_epochs = iter, analytic = True)
    Y_pred_ana = model_basic.predict(X_test)
    model_basic.train(generator_builder if use_generator_builder else (X_train, Y_train), n_epochs = iter, analytic = False)
    print('basic analytic', model_basic.error)
    Y_pred_basic = model_basic.predict(X_test)
    print('basic grad. descent', model_basic.error)

    model_enet = decontamination.Regression_ElasticNet(10, dtype = np.float32, rho = rho, l1_ratio = l1_ratio, alpha = 0.01, tolerance = None)
    model_enet.train(generator_builder if use_generator_builder else (X_train, Y_train), n_epochs = iter, soft_thresholding = False)
    Y_pred_enet = model_enet.predict(X_test)
    print('elastic net', model_enet.error)
    print(model_enet.weights)
    model_enet.train(generator_builder if use_generator_builder else (X_train, Y_train), n_epochs = iter, soft_thresholding = True)
    Y_pred_enet_soft = model_enet.predict(X_test)
    print('elastic net soft', model_enet.error)
    print(model_enet.weights)

    model_sklearn = SklearnElasticNet(alpha = rho, l1_ratio = l1_ratio, max_iter = iter)
    model_sklearn.fit(X_train, Y_train)
    Y_pred_sklearn = model_sklearn.predict(X_test)

    return Y_pred_ana, Y_pred_basic, Y_pred_enet, Y_pred_enet_soft, Y_pred_sklearn

########################################################################################################################

def test_regression():

    Y_pred_ana_nmb, Y_pred_basic_nmb, Y_pred_enet_nmb, Y_pred_enet_soft_nmb, Y_pred_sklearn_nmb = do_regressions(True)

    if True:

        plt.figure(figsize = (14, 7))
        plt.plot(Y_test, label = 'Actual values', color = 'yellow', marker = None)
        plt.plot(Y_pred_ana_nmb, label = 'decontamination ana model predictions', color = 'red', linestyle = '--', marker = None)
        plt.plot(Y_pred_basic_nmb, label = 'decontamination basic model predictions', color = 'orange', linestyle = '-.', marker = None)
        plt.plot(Y_pred_enet_nmb, label = 'decontamination enet model predictions', color = 'blue', linestyle = '--', marker = None)
        plt.plot(Y_pred_enet_soft_nmb, label = 'decontamination enet soft model predictions', color = 'cyan', linestyle = '-.', marker = None)
        plt.plot(Y_pred_sklearn_nmb, label = 'sklearn model predictions', color = 'green', linestyle = '--', marker = None)
        plt.xlabel('Test Sample Index')
        plt.ylabel('Output Value')
        plt.legend()
        plt.show()

    if True:

        Y_pred_ana_mb, Y_pred_basic_mb, Y_pred_enet_mb, Y_pred_enet_soft_mb, Y_pred_sklearn_mb = do_regressions(False)

        plt.figure(figsize = (14, 7))
        plt.plot((Y_pred_basic_nmb - Y_pred_basic_mb) / Y_pred_basic_nmb, label = 'decontamination ana model predictions', color = 'red', linestyle = '--', marker = 'x')
        plt.plot((Y_pred_basic_nmb - Y_pred_basic_mb) / Y_pred_basic_nmb, label = 'decontamination basic model predictions', color = 'red', linestyle = '-.', marker = 'x')
        plt.plot((Y_pred_enet_nmb - Y_pred_enet_mb) / Y_pred_enet_nmb, label = 'decontamination enet model predictions', color = 'blue', linestyle = '--', marker = 'x')
        plt.plot((Y_pred_enet_soft_nmb - Y_pred_enet_soft_mb) / Y_pred_enet_soft_nmb, label = 'decontamination enet soft model predictions', color = 'blue', linestyle = '-.', marker = 'x')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Relative difference')
        plt.legend()
        plt.show()

########################################################################################################################

if __name__ == '__main__':

    test_regression()

########################################################################################################################
