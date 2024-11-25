#!/usr/bin/env python3 -m pytest
# -*- coding: utf-8 -*-
########################################################################################################################

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

########################################################################################################################

import decontamination

import numpy as np

########################################################################################################################

def test_selection():

    dtype = [
        ('foo', np.float32),
        ('bar', np.float32),
        ('qux', np.float32),
    ]

    table = np.array([
        (1.0, 2.0, 3.5),
        (4.0, 5.8, 6.1),
        (1.0, 4.0, 3.2),
    ], dtype = dtype)

    expression, mask = decontamination.Selection.filter_table('', table)
    assert expression == ''
    assert np.array_equal(mask, [True, True, True])

    expression, mask = decontamination.Selection.filter_table('foo == 1.0 && (bar == 4.0 || qux >= 3.5)', table)
    assert expression == '(foo == 1.0) && ((bar == 4.0) || (qux >= 3.5))'
    assert np.array_equal(mask, [True, False, True])

    expression, mask = decontamination.Selection.filter_table('~(foo == 1.0 && (bar == 4.0 || qux >= 3.5))', table)
    assert expression == '~((foo == 1.0) && ((bar == 4.0) || (qux >= 3.5)))'
    assert np.array_equal(mask, [False, True, False])

    expression, mask = decontamination.Selection.filter_table('isfinite foo', table)
    assert expression == 'isfinite foo'
    assert np.array_equal(mask, [True, True, True])

########################################################################################################################
