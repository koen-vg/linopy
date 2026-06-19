#!/usr/bin/env python3
"""
Tests for the GLADE-fork extensions to linopy.

These cover two features carried on top of upstream linopy for the GLADE
food-systems model:

* ``Model.solve(..., calculate_fixed_duals=True)`` -- recover dual values for a
  MIP by re-solving the problem with all integer variables fixed to their
  optimal values. Honoured by the Highs and Gurobi direct backends.
* ``Model._mip_start = (n_entries, col_indices, col_values)`` -- inject a MIP
  starting solution, passed to ``highspy.setSolution`` / ``gurobipy`` var starts.
"""

import pandas as pd
import pytest

from linopy import Model, solvers

SOLVERS = ["highs", "gurobi"]


def _simple_milp() -> Model:
    """A small MILP with a binding constraint whose LP relaxation has a dual."""
    m = Model()
    lower = pd.Series(0, range(5))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(coords=x.coords, name="y", binary=True)
    m.add_constraints(x + y >= 10, name="con")
    m.add_objective(2 * x + y)
    return m


def _require(solver_name: str) -> None:
    if solver_name not in set(solvers.available_solvers):
        pytest.skip(f"{solver_name} is not installed")


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_calculate_fixed_duals(solver_name: str) -> None:
    """``calculate_fixed_duals=True`` yields finite MIP duals."""
    _require(solver_name)

    m = _simple_milp()
    status, condition = m.solve(solver_name=solver_name, calculate_fixed_duals=True)

    assert status == "ok"
    assert condition == "optimal"
    assert m.dual is not None
    assert len(m.dual.data_vars) > 0
    # With fixed integer variables the recovered duals must be finite.
    for var_name in m.dual.data_vars:
        assert not m.dual[var_name].isnull().all()


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_calculate_fixed_duals_default_off(solver_name: str) -> None:
    """The fixed-dual recomputation must not change the optimal objective."""
    _require(solver_name)

    m_off = _simple_milp()
    status, condition = m_off.solve(solver_name=solver_name)
    assert status == "ok"
    assert condition == "optimal"

    m_on = _simple_milp()
    m_on.solve(solver_name=solver_name, calculate_fixed_duals=True)
    assert float(m_on.objective.value) == pytest.approx(float(m_off.objective.value))


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_calculate_fixed_duals_lp_noop(solver_name: str) -> None:
    """For a pure LP the flag is a no-op and ordinary duals are returned."""
    _require(solver_name)

    m = Model()
    a = m.add_variables(lower=0, name="a")
    m.add_constraints(a >= 3, name="ca")
    m.add_objective(a)
    status, condition = m.solve(solver_name=solver_name, calculate_fixed_duals=True)
    assert status == "ok"
    assert condition == "optimal"
    assert float(m.constraints["ca"].dual) == pytest.approx(1.0)


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_mip_start(solver_name: str) -> None:
    """A model-level MIP start is accepted and yields the optimal solution."""
    _require(solver_name)

    # Baseline solve to obtain the optimal objective.
    baseline = _simple_milp()
    baseline.solve(solver_name=solver_name)
    opt = float(baseline.objective.value)

    m = _simple_milp()
    # Binary variables y occupy columns 5..9 (x occupies 0..4); start them at 1.
    indices = list(range(5, 10))
    values = [1.0] * len(indices)
    m._mip_start = (len(indices), indices, values)
    status, condition = m.solve(solver_name=solver_name)

    assert status == "ok"
    assert condition == "optimal"
    assert float(m.objective.value) == pytest.approx(opt)


def test_mip_start_default_none() -> None:
    """A freshly constructed model has no MIP start set."""
    m = Model()
    assert m._mip_start is None
