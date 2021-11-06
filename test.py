from copy import deepcopy

import numpy as np
from yaml import safe_load

from wildfiresim.optimization import NelderMead, OptVar, Torczon
from wildfiresim.simulation import Forest, Simulation, SimulationState


def cost_function(x: OptVar, simulation: Simulation, initial_state: SimulationState) -> float:
    """initial state before cutting down the trees"""
    xmin, xmax, ymin, ymax = x
    forest = simulation.forest

    final_state = simulation.simulate(initial_state=initial_state, x=x)

    forest_cost = initial_state.get_fuel_amount(dx=forest.dx) - final_state.get_fuel_amount(dx=forest.dx)
    area_cost = abs(xmax - xmin) * abs(ymax - ymin)
    position_cost = max(0, 0.2 - ymin)

    return forest_cost + 10 * area_cost + 100 * position_cost


def main():
    seed = 1

    with open("params.yml", "r") as parfile:
        params = safe_load(parfile)

    forest = Forest(**params["forest"])
    simulation = Simulation(forest=forest, **params["simulation"])
    initial_state = SimulationState(X=forest.X, Y=forest.Y, **params["initial_state"], rng=1)
    initial_simplex = np.random.default_rng(seed).random((5, 4))

    # Simulation
    initial_state
    simulation.simulate(initial_state=initial_state, track_state=True, verbose=True)
    simulation.animate(nb_frames=100, title="Simulation sans coupe-feu")

    # Optimisation
    funckwargs = {"simulation": simulation, "initial_state": initial_state}
    optimization_results = []
    for optimizer_name, OptimizerClass in (("nelder_mead", NelderMead), ("torczon", Torczon)):
        print(optimizer_name)
        optimizer = OptimizerClass(**{**params["optimizer"], **params[optimizer_name]})
        x, f, _ = optimizer.minimize(
            function=cost_function, initial_simplex=initial_simplex, funckwargs=funckwargs, track_cost=True, verbose=True
        )
        optimizer.animate(initial_state=initial_state, simulation=simulation, title=optimizer_name, show_final_state=True)
        optimization_results.append((optimizer_name, x, f))

    for optimizer_name, x, f in optimization_results:
        initial_state_opt = deepcopy(initial_state)
        initial_state_opt.cut_trees(forest.X, forest.Y, *x)
        simulation.simulate(initial_state=initial_state_opt, track_state=True)
        simulation.animate(nb_frames=100, title=f"Solution de {optimizer_name}\nx = {np.round(x, 2)}, {f =:.2f}")


if __name__ == "__main__":
    main()
