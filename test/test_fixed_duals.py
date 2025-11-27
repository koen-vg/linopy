#!/usr/bin/env python3
"""
Tests for calculate_fixed_duals functionality.
"""

import pytest

from linopy import Model, solvers


def create_test_milp() -> Model:
    """
    Create a test MILP model.

    Min    f  = -3x_0 - 2x_1 - x_2
    s.t.          x_0 +  x_1 + x_2 <=  7
                 4x_0 + 2x_1 + x_2  = 12
                 x_0 >=0; x_1 >= 0; x_2 binary
    """
    m = Model()

    # Variables
    x0 = m.add_variables(lower=0, name="x0")
    x1 = m.add_variables(lower=0, name="x1")
    x2 = m.add_variables(name="x2", binary=True)

    # Constraints
    m.add_constraints(x0 + x1 + x2 <= 7, name="inequality")
    m.add_constraints(4 * x0 + 2 * x1 + x2 == 12, name="equality")

    # Objective
    m.add_objective(-3 * x0 - 2 * x1 - x2)

    return m


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_calculate_fixed_duals() -> None:
    """Test calculate_fixed_duals with Gurobi solver (file-based API)."""
    m = create_test_milp()

    # Solve with calculate_fixed_duals=True
    status, condition = m.solve(solver_name="gurobi", calculate_fixed_duals=True)

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    assert m.dual is not None

    # Check that dual values are present (m.dual is an xarray Dataset)
    assert len(m.dual.data_vars) > 0

    # Check that dual values are not all NaN
    for var_name in m.dual.data_vars:
        assert not m.dual[var_name].isnull().all()


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_calculate_fixed_duals_false() -> None:
    """Test that calculate_fixed_duals=False (default) works for Gurobi solver."""
    m = create_test_milp()

    # Solve without calculate_fixed_duals (default is False)
    status, condition = m.solve(solver_name="gurobi", calculate_fixed_duals=False)

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    # For MIP without fixed duals, dual values might not be available
    # but the solution should still be valid


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_calculate_fixed_duals_direct_api() -> None:
    """Test calculate_fixed_duals with Gurobi direct API."""
    m = create_test_milp()

    # Solve with direct API and calculate_fixed_duals=True
    status, condition = m.solve(
        solver_name="gurobi", io_api="direct", calculate_fixed_duals=True
    )

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    assert m.dual is not None

    # Check that dual values are present
    assert len(m.dual.data_vars) > 0

    # Check that dual values are not all NaN
    for var_name in m.dual.data_vars:
        assert not m.dual[var_name].isnull().all()


@pytest.mark.skipif(
    "highs" not in set(solvers.available_solvers), reason="Highs is not installed"
)
def test_highs_calculate_fixed_duals() -> None:
    """Test calculate_fixed_duals with Highs solver (file-based API)."""
    m = create_test_milp()

    # Solve with calculate_fixed_duals=True
    status, condition = m.solve(solver_name="highs", calculate_fixed_duals=True)

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    assert m.dual is not None

    # Check that dual values are present (m.dual is an xarray Dataset)
    assert len(m.dual.data_vars) > 0

    # Check that dual values are not all NaN
    for var_name in m.dual.data_vars:
        assert not m.dual[var_name].isnull().all()


@pytest.mark.skipif(
    "highs" not in set(solvers.available_solvers), reason="Highs is not installed"
)
def test_highs_calculate_fixed_duals_false() -> None:
    """Test that calculate_fixed_duals=False (default) works for Highs solver."""
    m = create_test_milp()

    # Solve without calculate_fixed_duals (default is False)
    status, condition = m.solve(solver_name="highs", calculate_fixed_duals=False)

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    # Solution should be valid even without fixed duals


@pytest.mark.skipif(
    "highs" not in set(solvers.available_solvers), reason="Highs is not installed"
)
def test_highs_calculate_fixed_duals_direct_api() -> None:
    """Test calculate_fixed_duals with Highs direct API."""
    m = create_test_milp()

    # Solve with direct API and calculate_fixed_duals=True
    status, condition = m.solve(
        solver_name="highs", io_api="direct", calculate_fixed_duals=True
    )

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    assert m.dual is not None

    # Check that dual values are present
    assert len(m.dual.data_vars) > 0

    # Check that dual values are not all NaN
    for var_name in m.dual.data_vars:
        assert not m.dual[var_name].isnull().all()
