.. _fixed-duals:

Fixed Duals for Mixed-Integer Programs
=======================================

Dual values (shadow prices) are normally only available for continuous linear programs. For mixed-integer programs (MIPs), duals are not well-defined due to the discrete nature of integer variables. However, linopy supports computing approximate dual values for MIPs through a technique called **fixed duals**.

.. contents::
   :local:
   :depth: 2

Overview
--------

Fixed duals provide sensitivity information for mixed-integer programs by:

1. Solving the MIP to optimality
2. Fixing all integer variables to their optimal values
3. Resolving the resulting continuous LP
4. Extracting dual values from the fixed LP

This approach is particularly useful for:

- Understanding constraint sensitivity at a specific integer solution
- Post-optimality analysis of MIP models
- Approximating marginal costs in production planning with discrete decisions
- Sensitivity analysis in facility location and network design problems

**Important limitation:** Fixed duals represent sensitivity only at the specific integer solution found. They do not capture the general MIP sensitivity, which would require analyzing the entire branch-and-bound tree.

Solver Support
--------------

Fixed duals are currently supported by:

- **Gurobi**: Uses the ``Model.fixed()`` method
- **Highs**: Uses the ``Highs.getFixedLp()`` method

Other solvers may be added in future versions.

Basic Usage
-----------

To compute fixed duals for a MIP model, pass ``calculate_fixed_duals=True`` to the solve method:

.. code-block:: python

    import linopy
    import pandas as pd

    # Create a simple MILP model
    m = linopy.Model()

    x = m.add_variables(lower=0, name="x")
    y = m.add_variables(lower=0, name="y", binary=True)

    m.add_constraints(x + y >= 5, name="min_total")
    m.add_objective(2 * x + 3 * y)

    # Solve with fixed duals enabled
    m.solve(solver_name="gurobi", calculate_fixed_duals=True)

    # Access dual values
    print(m.dual)

Without ``calculate_fixed_duals=True``, the ``m.dual`` attribute would be empty or undefined for MIP models.
