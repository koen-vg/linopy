#!/usr/bin/env python3
"""
Created on Tue Jan 28 09:03:35 2025.

@author: sid
"""

from pathlib import Path

import pytest
from test_io import model  # noqa: F401

from linopy import Model, solvers

free_mps_problem = """NAME               sample_mip
ROWS
 N  obj
 G  c1
 L  c2
 E  c3
COLUMNS
    col1        obj       5
    col1        c1        2
    col1        c2        4
    col1        c3        1
    MARK0000  'MARKER'                 'INTORG'
    colu2        obj       3
    colu2        c1        3
    colu2        c2        2
    colu2        c3        1
    col3        obj       7
    col3        c1        4
    col3        c2        3
    col3        c3        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS_V     c1        12
    RHS_V     c2        15
    RHS_V     c3        6
BOUNDS
 UP BOUND     col1        4
 UI BOUND     colu2        3
 UI BOUND     col3        5
ENDATA
"""


@pytest.mark.parametrize("solver", set(solvers.available_solvers))
def test_free_mps_solution_parsing(solver: str, tmp_path: Path) -> None:
    try:
        solver_enum = solvers.SolverName(solver.lower())
        solver_class = getattr(solvers, solver_enum.name)
    except ValueError:
        raise ValueError(f"Solver '{solver}' is not recognized")

    # Write the MPS file to the temporary directory
    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)

    # Create a solution file path in the temporary directory
    sol_file = tmp_path / "solution.sol"

    s = solver_class()
    result = s.solve_problem(problem_fn=mps_file, solution_fn=sol_file)

    assert result.status.is_ok
    assert result.solution.objective == 30.0


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_environment_with_dict(model: Model, tmp_path: Path) -> None:  # noqa: F811
    gurobi = solvers.Gurobi()

    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)
    sol_file = tmp_path / "solution.sol"

    log1_file = tmp_path / "gurobi1.log"
    result = gurobi.solve_problem(
        problem_fn=mps_file, solution_fn=sol_file, env={"LogFile": str(log1_file)}
    )

    assert result.status.is_ok
    assert log1_file.exists()

    log2_file = tmp_path / "gurobi2.log"
    gurobi.solve_problem(
        model=model, solution_fn=sol_file, env={"LogFile": str(log2_file)}
    )
    assert result.status.is_ok
    assert log2_file.exists()


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_environment_with_gurobi_env(model: Model, tmp_path: Path) -> None:  # noqa: F811
    import gurobipy as gp

    gurobi = solvers.Gurobi()

    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)
    sol_file = tmp_path / "solution.sol"

    log1_file = tmp_path / "gurobi1.log"

    with gp.Env(params={"LogFile": str(log1_file)}) as env:
        result = gurobi.solve_problem(
            problem_fn=mps_file, solution_fn=sol_file, env=env
        )

    assert result.status.is_ok
    assert log1_file.exists()

    log2_file = tmp_path / "gurobi2.log"
    with gp.Env(params={"LogFile": str(log2_file)}) as env:
        gurobi.solve_problem(model=model, solution_fn=sol_file, env=env)
    assert result.status.is_ok
    assert log2_file.exists()


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_calculate_fixed_duals() -> None:
    """Test that calculate_fixed_duals option works for Gurobi solver with MIP."""
    import pandas as pd

    m = Model()

    # Create a simple MILP model
    lower = pd.Series(0, range(5))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(coords=x.coords, name="y", binary=True)

    m.add_constraints(x + y >= 10, name="con")

    m.add_objective(2 * x + y)

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
    import pandas as pd

    m = Model()

    # Create a simple MILP model
    lower = pd.Series(0, range(5))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(coords=x.coords, name="y", binary=True)

    m.add_constraints(x + y >= 10, name="con")

    m.add_objective(2 * x + y)

    # Solve without calculate_fixed_duals (default is False)
    status, condition = m.solve(solver_name="gurobi", calculate_fixed_duals=False)

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    # For MIP without fixed duals, dual values might not be available
    # but the solution should still be valid


@pytest.mark.skipif(
    "highs" not in set(solvers.available_solvers), reason="Highs is not installed"
)
def test_highs_calculate_fixed_duals() -> None:
    """Test that calculate_fixed_duals option works for Highs solver with MILP."""
    import pandas as pd

    m = Model()

    # Create a simple MILP model
    lower = pd.Series(0, range(5))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(coords=x.coords, name="y", binary=True)

    m.add_constraints(x + y >= 10, name="con")

    m.add_objective(2 * x + y)

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
    import pandas as pd

    m = Model()

    # Create a simple MILP model
    lower = pd.Series(0, range(5))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(coords=x.coords, name="y", binary=True)

    m.add_constraints(x + y >= 10, name="con")

    m.add_objective(2 * x + y)

    # Solve without calculate_fixed_duals (default is False)
    status, condition = m.solve(solver_name="highs", calculate_fixed_duals=False)

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    # Solution should be valid even without fixed duals


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_calculate_fixed_duals_with_direct_api() -> None:
    """Test calculate_fixed_duals with direct API (io_api='direct')."""
    import pandas as pd

    m = Model()

    # Create a simple MILP model
    lower = pd.Series(0, range(3))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, 5, name="y", integer=True)

    m.add_constraints(x + y >= 5, name="con")

    m.add_objective(x + 2 * y)

    # Solve with direct API and calculate_fixed_duals=True
    status, condition = m.solve(
        solver_name="gurobi", io_api="direct", calculate_fixed_duals=True
    )

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    assert m.dual is not None
    assert len(m.dual.data_vars) > 0
    # Check that dual values are not all NaN
    for var_name in m.dual.data_vars:
        assert not m.dual[var_name].isnull().all()


@pytest.mark.skipif(
    "highs" not in set(solvers.available_solvers), reason="Highs is not installed"
)
def test_highs_calculate_fixed_duals_with_direct_api() -> None:
    """Test calculate_fixed_duals with direct API (io_api='direct')."""
    import pandas as pd

    m = Model()

    # Create a simple MILP model
    lower = pd.Series(0, range(3))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, 5, name="y", integer=True)

    m.add_constraints(x + y >= 5, name="con")

    m.add_objective(x + 2 * y)

    # Solve with direct API and calculate_fixed_duals=True
    status, condition = m.solve(
        solver_name="highs", io_api="direct", calculate_fixed_duals=True
    )

    assert status == "ok"
    assert condition == "optimal"
    assert m.solution is not None
    assert m.dual is not None
    assert len(m.dual.data_vars) > 0
    # Check that dual values are not all NaN
    for var_name in m.dual.data_vars:
        assert not m.dual[var_name].isnull().all()
