.. _fixed-duals:

Fixed Duals and MIP Starts
==========================

This page documents two GLADE-fork extensions that help when working with
mixed-integer programs (MIPs): recovering dual values from a fixed LP, and
injecting a MIP starting solution.

.. contents::
   :local:
   :depth: 2

Fixed Duals for Mixed-Integer Programs
--------------------------------------

Dual values (shadow prices) are normally only available for continuous linear
programs. For mixed-integer programs (MIPs), duals are not well-defined due to
the discrete nature of integer variables. linopy supports computing approximate
dual values for MIPs through a technique called **fixed duals**.

Fixed duals provide sensitivity information for mixed-integer programs by:

1. Solving the MIP to optimality
2. Fixing all integer variables to their optimal values
3. Resolving the resulting continuous LP
4. Extracting dual values from the fixed LP

This approach is particularly useful for:

- Understanding constraint sensitivity at a specific integer solution
- Post-optimality analysis of MIP models
- Approximating marginal costs in production planning with discrete decisions

**Important limitation:** Fixed duals represent sensitivity only at the specific
integer solution found. They do not capture the general MIP sensitivity, which
would require analyzing the entire branch-and-bound tree.

Solver Support
~~~~~~~~~~~~~~

Fixed duals are currently supported by:

- **Gurobi**: uses the ``Model.fixed()`` method
- **HiGHS**: uses the ``Highs.getFixedLp()`` method

The flag is a no-op for pure LPs (ordinary duals are returned) and for solvers
that do not implement it.

Basic Usage
~~~~~~~~~~~

To compute fixed duals for a MIP model, pass ``calculate_fixed_duals=True`` to
the solve method:

.. code-block:: python

    import linopy

    m = linopy.Model()

    x = m.add_variables(lower=0, name="x")
    y = m.add_variables(lower=0, name="y", binary=True)

    m.add_constraints(x + y >= 5, name="min_total")
    m.add_objective(2 * x + 3 * y)

    # Solve with fixed duals enabled
    m.solve(solver_name="highs", calculate_fixed_duals=True)

    # Access dual values
    print(m.dual)

Without ``calculate_fixed_duals=True``, the ``m.dual`` values for a MIP are
typically zero or undefined.

MIP Starts
----------

A MIP starting solution can speed up the branch-and-bound search by handing the
solver a known feasible (or partial) assignment of the integer variables. Set
the ``Model._mip_start`` attribute before solving:

.. code-block:: python

    # (n_entries, column_indices, column_values)
    m._mip_start = (len(indices), indices, values)
    m.solve(solver_name="highs")

The column indices refer to variable columns in solver order (matching
``model.matrices.vlabels``). The MIP start is forwarded to
``highspy.Highs.setSolution`` for HiGHS and set as ``Var.Start`` values for
Gurobi. It is ignored by solvers that do not support warm starting from a
solution vector.
