#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

from sklearn.linear_model import ElasticNet as SklearnElasticNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import decontamination

import numpy as np

import matplotlib.pyplot as plt

########################################################################################################################

rnd = np.random.default_rng(seed = 0)

########################################################################################################################

true_coef = np.array([1.5, -2.0, 3.0, 0, 0, 0, 0, 0, 0, 0])

X_train = rnd.standard_normal((160, 10))
Y_train = np.dot(X_train, true_coef) + rnd.standard_normal(160) * 0.5

X_test = rnd.standard_normal((40, 10))
Y_test = np.dot(X_test, true_coef) + rnd.standard_normal(40) * 0.5

########################################################################################################################

def test_regression():

    model_sklearn = SklearnElasticNet(alpha = 0.1, l1_ratio = 0.5, max_iter = 1000)
    model_sklearn.fit(X_train, Y_train)
    Y_pred_sklearn = model_sklearn.predict(X_test)

    model_basic = decontamination.Regression_Basic(10, dtype = np.float32, alpha = 0.01, tolerance = None)
    model_basic.train((X_train, Y_train), n_epochs = 1000, analytic = True)
    Y_pred_ana = model_basic.predict(X_test)
    model_basic.train((X_train, Y_train), n_epochs = 1000, analytic = False)
    Y_pred_basic = model_basic.predict(X_test)

    model_enet = decontamination.Regression_ElasticNet(10, dtype = np.float32, rho = 0.1, l1_ratio = 0.5, alpha = 0.01, tolerance = None)
    model_enet.train((X_train, Y_train), n_epochs = 1000)
    Y_pred_enet = model_enet.predict(X_test)

    plt.figure(figsize = (14, 7))
    plt.plot(Y_test, label = 'Actual values', color = 'blue', marker = 'o')
    plt.plot(Y_pred_sklearn, label = 'sklearn model predictions', color = 'green', linestyle = '--', marker = '+')
    plt.plot(Y_pred_ana, label = 'decontamination ana model predictions', color = 'yellow', linestyle = '--', marker = 'x')
    plt.plot(Y_pred_basic + 0.01, label = 'decontamination basic model predictions', color = 'red', linestyle = '--', marker = 'x')
    plt.plot(Y_pred_enet, label = 'decontamination enet model predictions', color = 'blue', linestyle = '--', marker = 'x')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()

########################################################################################################################

if __name__ == '__main__':

    test_regression()

########################################################################################################################
