.. _fixed-duals:

Duals for Mixed-Integer Programs by Fixing Integer Variables
========================================================================

Dual values (shadow prices) are only well-defined for continuous linear programs. For mixed-integer programs (MIPs), duals are not well-defined due to the discrete nature of integer variables. However, you can still get information on dual values "around the optimum" by fixing all integer variables to their optimal values and considering the dual values of the resulting LP.

Specifically, linopy has built-in support for the following:

1. Solving the MIP to optimality
2. Fixing all integer variables to their optimal values
3. Resolving the resulting continuous LP
4. Extracting dual values from the fixed LP

**Important limitation:** Fixed duals represent sensitivity only at the specific integer solution found. They do not capture the general MIP sensitivity, which would require analyzing the entire branch-and-bound tree.

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

Solver Support
--------------

Fixed duals are currently supported by:

- **Gurobi**: Uses the ``Model.fixed()`` method
- **Highs**: Uses the ``Highs.getFixedLp()`` method

Other solvers may be added in future versions.
