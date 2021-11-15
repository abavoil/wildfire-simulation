from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from yaml import safe_load

from main import cost_function, make_file_path, print_msg
from wildfiresim.optimization import (
    CountedFunction,
    NelderMead,
    OptimizationState,
    SimplexOptimizer,
    Torczon,
)
from wildfiresim.simulation import Forest, Simulation, SimulationState


def test_x0():
    seed = 1
    param_file = "params.yml"
    with open(param_file, "r") as parfile:
        params = safe_load(parfile)

    dist = 0.1
    x0_values = [[0.1, 0.3, 0.2, 0.4], [0.4, 0.8, 0.4, 0.6], [0.4, 0.6, 0.6, 0.7]]
    opt_names = ("Nelder-Mead", "Torczon")
    opt_classes = (NelderMead, Torczon)

    forest = Forest(**params["forest"])
    simulation = Simulation(forest=forest, **params["simulation"])
    initial_state = SimulationState.create_initial_state(X=forest.X, Y=forest.Y, **params["initial_state"], rng=seed)
    funckwargs = {"simulation": simulation, "initial_state": initial_state}

    cost_historic = dict.fromkeys(map(str, x0_values), dict.fromkeys(opt_names))
    cost_historic = {str(x0): {optimizer_name: None for optimizer_name in opt_names} for x0 in x0_values}

    for x0 in x0_values:
        print_msg(f"{x0=}")

        for optimizer_name, OptimizerClass in zip(opt_names, opt_classes):
            print_msg(optimizer_name, indent=1)

            counted_function = CountedFunction(function=cost_function, funckwargs=funckwargs)
            initial_opt_state = OptimizationState.create_initial_state(counted_function=counted_function, x0=np.array(x0), dist=dist)
            optimizer: SimplexOptimizer = OptimizerClass(**{**params["optimizer"], **params[optimizer_name]})
            optimizer.minimize(counted_function=counted_function, initial_opt_state=initial_opt_state, track_cost=True)
            optimizer.animate(
                initial_state=initial_state,
                simulation=simulation,
                title=f"Solution of {optimizer_name}\n{x0=} and d={dist}",
                show=False,
                filepath=make_file_path("img_out", f"{optimizer_name} solution {x0=} d={dist}", "mp4"),
            )
            cost_historic[str(x0)][optimizer_name] = deepcopy(optimizer.get_cost_historic())  # type:ignore

    print(cost_historic)
    for x0 in x0_values:
        plt.subplots(figsize=(8, 6))
        for optimizer_name in opt_names:
            plt.plot(*cost_historic[str(x0)][optimizer_name], label=optimizer_name)  # type:ignore
        plt.legend()
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel("Number of calls")
        plt.ylabel("Best value")
        plt.title(f"x0={x0}\nd={dist}")
        plt.savefig(make_file_path("img_out", f"benchmark {x0=}", "png"))


def test_dist():
    seed = 1
    param_file = "params.yml"
    with open(param_file, "r") as parfile:
        params = safe_load(parfile)

    dist_values = [0.01, 0.1, 0.3]
    x0 = [0.45, 0.85, 0.4, 0.6]
    opt_names = ("Nelder-Mead", "Torczon")
    opt_classes = (NelderMead, Torczon)
    # opt_colors = plt.get_cmap("hsv")(np.linspace(0, 1, len(opt_names), endpoint=False))

    forest = Forest(**params["forest"])
    simulation = Simulation(forest=forest, **params["simulation"])
    initial_state = SimulationState.create_initial_state(X=forest.X, Y=forest.Y, **params["initial_state"], rng=seed)
    funckwargs = {"simulation": simulation, "initial_state": initial_state}

    cost_historic = dict.fromkeys(dist_values, dict.fromkeys(opt_names).copy())
    cost_historic = {dist: {optimizer_name: None for optimizer_name in opt_names} for dist in dist_values}

    for dist in dist_values:
        print_msg(f"{dist=}")

        for optimizer_name, OptimizerClass in zip(opt_names, opt_classes):
            print_msg(optimizer_name, indent=1)

            counted_function = CountedFunction(function=cost_function, funckwargs=funckwargs)
            initial_opt_state = OptimizationState.create_initial_state(counted_function=counted_function, x0=np.array(x0), dist=dist)
            optimizer: SimplexOptimizer = OptimizerClass(**{**params["optimizer"], **params[optimizer_name]})
            optimizer.minimize(counted_function=counted_function, initial_opt_state=initial_opt_state, track_cost=True)
            optimizer.animate(
                initial_state=initial_state,
                simulation=simulation,
                title=f"Solution of {optimizer_name}\nd={dist} and {x0=}",
                show=False,
                filepath=make_file_path("img_out", f"{optimizer_name} solution d={dist} {x0=}", "mp4"),
            )
            cost_historic[dist][optimizer_name] = deepcopy(optimizer.get_cost_historic())  # type:ignore

    print(cost_historic)
    for dist in dist_values:
        plt.subplots(figsize=(8, 6))
        for optimizer_name in opt_names:
            plt.step(*cost_historic[dist][optimizer_name], label=optimizer_name)  # type: ignore
        plt.legend()
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.xlabel("Number of calls")
        plt.ylabel("Best value")
        plt.title(f"d={dist}\n{x0=}")
        plt.savefig(make_file_path("img_out", f"benchmark d={dist}", "png"))


if __name__ == "__main__":
    test_x0()
    test_dist()
