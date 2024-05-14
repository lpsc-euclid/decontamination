#!/usr/bin/env python3 -m pytest
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

from sklearn.model_selection import train_test_split

from sklearn.linear_model import ElasticNet as SklearnElasticNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import decontamination

import numpy as np

import matplotlib.pyplot as plt

########################################################################################################################

true_coef = np.array([1.5, -2.0, 3.0, 0, 0, 0, 0, 0, 0, 0])

rnd = np.random.default_rng(seed = 0)

X = rnd.standard_normal((200, 10))
Y = np.dot(X, true_coef) + rnd.standard_normal(200) * 0.5

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

########################################################################################################################

def test_centroids():

    model_sklearn = SklearnElasticNet(alpha = 0.1, l1_ratio = 0.5, max_iter = 1000)
    model_sklearn.fit(X_train, Y_train)
    Y_pred_sklearn = model_sklearn.predict(X_test)

    model_decontamination = decontamination.ElasticNet(10, dtype = np.float32, rho = 0.1, l1_ratio = 0.5, alpha = 0.01, tolerance = None)
    model_decontamination.train((X_train, Y_train), n_epochs = 1000)
    Y_pred_decontamination = model_decontamination.predict(X_test)

    plt.figure(figsize=(14, 7))
    plt.plot(Y_test, label='Actual values', color='blue', marker='o')
    plt.plot(Y_pred_sklearn, label='Sklearn model predictions', color='green', linestyle='--', marker='+')
    plt.plot(Y_pred_decontamination, label='VMPZ model predictions', color='red', linestyle='--', marker='x')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Output Value')
    plt.legend()
    plt.show()

########################################################################################################################
