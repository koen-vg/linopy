from benchmark_gurobipy import basic_model, knapsack_model

if snakemake.config["benchmark"] == "basic":
    model = basic_model
elif snakemake.config["benchmark"] == "knapsack":
    model = knapsack_model

n = int(snakemake.wildcards.N)
solver = snakemake.wildcards.solver
model(n, solver)
