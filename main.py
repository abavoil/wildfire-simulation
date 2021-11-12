from copy import deepcopy
from typing import List, Tuple

import numpy as np
from matplotlib.pyplot import show
from yaml import safe_load

from wildfiresim.optimization import (
    CountedFunction,
    NelderMead,
    OptimizationState,
    OptVar,
    Torczon,
)
from wildfiresim.simulation import Forest, Simulation, SimulationState
from wildfiresim.vprint import get_timestamp


def cost_function(x: OptVar, simulation: Simulation, initial_state: SimulationState) -> float:
    """initial state before cutting down the trees"""
    xmin, xmax, ymin, ymax = x
    forest = simulation.forest

    final_state = simulation.simulate(initial_state=initial_state, x=x)

    forest_cost = initial_state.get_fuel_amount(dx=forest.dx) - final_state.get_fuel_amount(dx=forest.dx)
    area_cost = abs(xmax - xmin) * abs(ymax - ymin)
    position_cost = max(0, 0.2 - ymin)

    return forest_cost + 10 * area_cost + 100 * position_cost


def best_firewall_anim(
    final_opt_state: OptimizationState, simulation: Simulation, initial_state: SimulationState, optimizer_name: str, show_plots: bool
):
    initial_state = deepcopy(initial_state)
    x = final_opt_state.get_best()
    value = final_opt_state.get_fbest()
    initial_state.cut_trees(simulation.forest.X, simulation.forest.Y, *x)
    simulation.simulate(initial_state=initial_state, track_state=True)
    simulation.animate(
        nb_frames=100,
        title=f"Solution de {optimizer_name}\nx = {np.round(x, 2)}, {value =:.2f}",
        filepath=make_file_path(f"{optimizer_name}'s solution simulation", "mp4", show_plots),
    )


def print_msg(msg: str, *args, indent: int = 0, **kwargs):
    tab = "\t"
    line_return = "\n"
    print(f"{line_return}{tab * indent}*** {msg} ***", *args, **kwargs)


def make_file_path(filename: str, ext: str, show_plots: bool):
    return None if show_plots else f"img_out/{filename} - {get_timestamp('%Y%m%dT%H%M%S')}.{ext}"


def main():
    show_plots = False
    seed = 1
    param_file = "params.yml"

    print_msg(f"Parsing parameters from {param_file}")
    with open(param_file, "r") as parfile:
        params = safe_load(parfile)

    forest = Forest(**params["forest"])
    simulation = Simulation(forest=forest, **params["simulation"])
    initial_state = SimulationState.create_initial_state(X=forest.X, Y=forest.Y, **params["initial_state"], rng=seed)

    # Simulation
    print_msg("Running the simulation without a firewall")
    simulation.simulate(initial_state=initial_state, track_state=True, verbose=True)
    simulation.animate(nb_frames=100, title="Simulation sans coupe-feu", filepath=make_file_path("blank simulation", "mp4", show_plots))

    # Optimisation
    print_msg("Running the optimization algorithms")
    funckwargs = {"simulation": simulation, "initial_state": initial_state}
    for optimizer_name, OptimizerClass in (("Nelder-Mead", NelderMead), ("Torczon", Torczon)):
        print_msg(optimizer_name, indent=1)
        optimizer = OptimizerClass(**{**params["optimizer"], **params[optimizer_name]})
        counted_function = CountedFunction(function=cost_function, funckwargs=funckwargs)
        initial_opt_state = OptimizationState.create_initial_state(counted_function=counted_function, ndim=4, rng=seed)
        final_opt_state, _ = optimizer.minimize(
            counted_function=counted_function, initial_opt_state=initial_opt_state, track_cost=True, verbose=True
        )

        optimizer.plot_cost(filepath=make_file_path(f"{optimizer_name} cost", "png", show_plots))
        optimizer.animate(
            initial_state=initial_state,
            simulation=simulation,
            title=optimizer_name,
            filepath=make_file_path(f"{optimizer_name} iterations", "mp4", show_plots),
        )
        best_firewall_anim(final_opt_state, simulation, initial_state, optimizer_name, show_plots=show_plots)


if __name__ == "__main__":
    main()
