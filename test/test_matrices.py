#!/usr/bin/env python3
"""
Created on Mon Oct 10 14:21:23 2022.

@author: fabian
"""

import numpy as np
import pandas as pd
import xarray as xr

from linopy import EQUAL, GREATER_EQUAL, Model


def test_basic_matrices() -> None:
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)

    assert m.matrices.A is not None
    assert m.matrices.A.shape == (*m.matrices.clabels.shape, *m.matrices.vlabels.shape)
    assert m.matrices.clabels.shape == m.matrices.sense.shape
    assert m.matrices.vlabels.shape == m.matrices.ub.shape
    assert m.matrices.vlabels.shape == m.matrices.lb.shape


def test_basic_matrices_masked() -> None:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    mask = pd.Series([True] * 8 + [False, False])
    y = m.add_variables(lower, name="y", mask=mask)

    m.add_constraints(x + y, GREATER_EQUAL, 10)

    m.add_constraints(y, GREATER_EQUAL, 0)

    m.add_objective(2 * x + y)

    assert m.matrices.A is not None
    assert m.matrices.A.shape == (*m.matrices.clabels.shape, *m.matrices.vlabels.shape)
    assert m.matrices.clabels.shape == m.matrices.sense.shape
    assert m.matrices.vlabels.shape == m.matrices.ub.shape
    assert m.matrices.vlabels.shape == m.matrices.lb.shape


def test_matrices_duplicated_variables() -> None:
    m = Model()

    x = m.add_variables(pd.Series([0, 0]), 1, name="x")
    y = m.add_variables(4, pd.Series([8, 10]), name="y")
    z = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T, name="z")
    m.add_constraints(x + x + y + y + z + z == 0)

    assert m.matrices.A is not None

    A = m.matrices.A.todense()
    assert A[0, 0] == 2
    assert np.isin(np.unique(np.array(A)), [0.0, 2.0]).all()


def test_matrices_float_c() -> None:
    # https://github.com/PyPSA/linopy/issues/200
    m = Model()

    x = m.add_variables(pd.Series([0, 0]), 1, name="x")
    m.add_objective(x * 1.5)

    c = m.matrices.c
    assert np.all(c == np.array([1.5, 1.5]))


def test_matrices_frozen_constraint_widened_after_adding_variables() -> None:
    """Regression test: ``freeze_constraints=True`` followed by variables and a
    constraint added afterwards must still assemble a consistent matrix.

    Each frozen constraint caches its CSR at the active-variable count present
    when it was frozen. Adding variables later grows that count, so the earlier
    constraint's CSR must be widened to the current count before the constraint
    blocks are stacked, otherwise the stack fails with an axis-1 dimension
    mismatch.
    """
    m = Model(freeze_constraints=True)
    x = m.add_variables(lower=0, coords=[range(100)], dims=["i"], name="x")
    m.add_constraints(x >= 1, name="c_base")
    # New variables and a constraint added after ``c_base`` has been frozen.
    y = m.add_variables(lower=0, coords=[range(50)], dims=["j"], name="y")
    m.add_constraints(y >= 2, name="c_after")
    m.add_objective(x.sum() + y.sum())

    n_vars = m.variables.label_index.n_active_vars
    assert m.matrices.A.shape[1] == n_vars
    assert m.matrices.A.shape[0] == m.matrices.clabels.shape[0]
