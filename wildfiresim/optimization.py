from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from tqdm.std import tqdm

from wildfiresim.no_history_exception import NoHistoryException
from wildfiresim.simulation import Simulation, SimulationState
from wildfiresim.vprint import vprint

OptVar = np.ndarray
Simplex = np.ndarray


class CountedFunction:
    def __init__(self, function: Callable[..., float], funckwargs: Dict[str, Any]):
        """
        function: cost function, (x, **funckwargs) -> cost
        funckwargs: keyword arguments to be passed to the function alongside x
        """
        self.f = function
        self.funckwargs = funckwargs
        self._nb_calls = 0

    def __call__(self, x: OptVar) -> float:
        self._nb_calls += 1
        return self.f(x, **self.funckwargs)

    def get_nb_calls(self) -> int:
        return self._nb_calls


@dataclass
class OptimizationState:
    nb_iter: int
    nb_calls: int  # not linked to CountedFunction.nb_calls, must be updated at each iteration
    simplex: Simplex
    counted_function: InitVar[CountedFunction]
    fsimplex: np.ndarray = field(init=False)

    def __post_init__(self, counted_function: CountedFunction):
        self.fsimplex = np.array([counted_function(x) for x in self.simplex])
        self.sort_simplex()

    def sort_simplex(self):
        # best: ind = 0
        # worst: ind = -1
        ind = np.argsort(self.fsimplex)
        self.simplex.take(ind, out=self.simplex, axis=0)
        self.fsimplex.take(ind, out=self.fsimplex)

    def get_best(self) -> OptVar:
        return self.simplex[0]

    def get_fbest(self) -> float:
        return self.fsimplex[0]

    def get_spread(self) -> float:
        return np.abs(self.simplex[1:] - self.get_best()).max()

    def get_fspread(self) -> float:
        return np.abs(self.fsimplex[1:] - self.get_fbest()).max()

    @staticmethod
    def create_initial_state(
        counted_function: CountedFunction, ndim: int, rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> OptimizationState:
        """
        rng: either None for non-reproducibility, int for seed, or Generator
        """

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        simplex = rng.uniform(0, 1, (ndim + 1, ndim))

        return OptimizationState(0, 0, simplex=simplex, counted_function=counted_function)

    @staticmethod
    def create_initial_state_nonrng(counted_function: CountedFunction, x0: OptVar, dist: float) -> OptimizationState:
        """right-angled isocele simplex around x0 with edges of length 2*dist"""
        ndim = x0.size
        simplex = np.repeat(x0.reshape(1, -1), ndim + 1, 0)
        simplex[0] -= dist
        for i in range(ndim):
            simplex[i + 1][i] += dist

        return OptimizationState(0, 0, simplex=simplex, counted_function=counted_function)


@dataclass
class SimplexOptimizer(ABC):
    xtol: float
    ftol: float
    maxiter: int
    maxfun: int
    history: Optional[List[OptimizationState]] = field(init=False)

    pattern: ClassVar[str] = "{:>10s}{:>10s}{:>10s}{:>10s}{:>35s}{:>25s}{:>10s}"  # for printing infos in a table format

    def minimize(
        self, counted_function: CountedFunction, initial_opt_state: OptimizationState, track_cost: bool = False, verbose: bool = False
    ) -> Tuple[OptimizationState, bool]:
        """
        find x that minimizes function(x, **funcargs)
        returns the simplex, its value, and a boolean (True if converged, False otherwise)
        """

        state = copy(initial_opt_state)

        self.history = [] if track_cost else None
        self._track_state(state)

        self._print_header(verbose=verbose)
        converged = False
        while counted_function.get_nb_calls() < self.maxfun and state.nb_iter < self.maxiter:
            state.nb_iter += 1
            state.nb_calls = counted_function.get_nb_calls()
            movement = self._minimization_step(counted_function, state)
            state.sort_simplex()

            self._print_row(
                state=state, movement=movement, verbose=verbose,
            )

            self._track_state(state)

            if state.get_spread() < self.xtol and state.get_fspread() < self.ftol:
                converged = True
                break

        # track last state
        self._track_state(state)
        if converged:
            reason = "xspread < self.xtol and fspread < self.ftol"
        elif counted_function.get_nb_calls() > self.maxfun:
            reason = "too many function calls"
        else:
            reason = "too many iterations"

        if not converged:
            print("*** WARNING ***\n did not converge \n*** END OF WARNING ***")

        vprint(f"Stopping minimization at nb_iter={state.nb_calls} ({converged=}) because {reason}.\n", verbose=verbose)
        return state, converged

    @abstractmethod
    def _minimization_step(self, counted_function: CountedFunction, state: OptimizationState) -> str:
        """
        modify simplex and fsimplex to advance one step
        When a vertex of the simplex is updated, its corresponding cost in fsimmplex must be updated aswell
        Returns the movement that has been executed to go to the next step
        """
        ...

    def _track_state(self, state: OptimizationState):
        if self.history is None:
            # tracking is not activated
            return
        if len(self.history) == 0 or np.any(self.history[-1].get_best() != state.get_best()):
            self.history.append(deepcopy(state))

    def _print_header(self, verbose: bool):
        vprint(
            self.pattern.format("nb_iter", "nb_calls", "x_spread", "f_spread", "movement (prev to curr)", "best", "f(best)"),
            verbose=verbose,
            timestamp=True,
        )

    def _print_row(self, state: OptimizationState, movement: str, verbose: bool):
        rounding = 2
        vprint(
            self.pattern.format(
                *map(
                    str,
                    (
                        state.nb_iter,
                        state.nb_calls,
                        round(state.get_spread(), rounding),
                        round(state.get_fspread(), rounding),
                        movement,
                        np.round(state.get_best(), rounding),
                        round(state.get_fbest(), rounding),
                    ),
                )
            ),
            verbose=verbose,
            timestamp=True,
        )

    def animate(self, initial_state: SimulationState, simulation: Simulation, title: str = "", filepath: Optional[str] = None) -> FuncAnimation:
        """
        initial_state: The initial state used to compute the cost function
        """

        if self.history is None:
            raise NoHistoryException()

        # compute all final states in advance
        sim_final_state_no_firewall = simulation.simulate(initial_state)  # first frame, without rectangle
        sim_final_states = [
            simulation.simulate(initial_state, opt_state.get_best()) for opt_state in tqdm(self.history, desc="Running simulations...")
        ]
        opt_final_state = self.history[-1]

        X, Y = simulation.forest.X, simulation.forest.Y

        fig, ax = plt.subplots()

        fuel = ax.pcolormesh(X, Y, sim_final_state_no_firewall.c, vmin=0, vmax=10, cmap=plt.cm.YlGn, shading="nearest")  # type: ignore
        fuel_cb = fig.colorbar(fuel, ax=ax)
        fuel_cb.set_label("fuel", loc="top")

        n_arrows = 6
        wind_ind = np.zeros(X.shape, dtype=bool)
        inds = slice((X.shape[0] % n_arrows) // 2, None, X.shape[0] // n_arrows)
        wind_ind[inds, inds] = True
        wind = ax.quiver(X[wind_ind], Y[wind_ind], simulation.forest.u[wind_ind], simulation.forest.v[wind_ind])

        firewall = Rectangle((0, 0), 0, 0, edgecolor="k", facecolor=(0, 0, 0, 0.1), lw=3)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax_text_info = ax.text(0.5, 0.05, "", bbox={"facecolor": "w", "alpha": 0.5, "pad": 5}, ha="center")
        title_pattern = "nb_call={nb_call}/{tot_nb_call}, cost={cost:.2f}\nx={x}"

        def init():
            fuel.set_array(sim_final_state_no_firewall.c.ravel())
            firewall.set_bounds(0, 0, 0, 0)
            ax.add_patch(firewall)
            ax_text_info.set_text("")
            return fuel, firewall, wind, ax_text_info

        def animate(frame: int, *_):
            opt_state = self.history[frame]  # type: ignore
            sim_state = sim_final_states[frame]
            fuel.set_array(sim_state.c.ravel())
            xmin, xmax, ymin, ymax = opt_state.get_best()
            firewall.set_bounds(xmin, ymin, xmax - xmin, ymax - ymin)
            ax_text_info.set_text(
                title_pattern.format(
                    nb_call=opt_state.nb_calls,
                    tot_nb_call=opt_final_state.nb_calls,
                    cost=opt_state.get_fbest(),
                    x=np.round(opt_state.get_best(), 2),
                )
            )
            return fuel, firewall, wind, ax_text_info

        fps = 2
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.history), interval=1000 / fps, repeat=True, repeat_delay=2000, blit=True)  # type: ignore

        if filepath is not None:
            plt.subplots(figsize=(8, 6))
            anim.save(filename=filepath, writer="ffmpeg", fps=fps)
        else:
            plt.show()

        return anim

    def plot_cost(self, title: str = "", filepath: Optional[str] = None):
        if self.history is None:
            raise NoHistoryException()

        call_counts = [state.nb_calls for state in self.history]  # type: ignore
        costs = [state.get_fbest() for state in self.history]  # type: ignore

        plt.step(call_counts, costs, where="post", marker="x", markeredgecolor="k", markersize=10)
        plt.title(title + "\nValue of the cost function against the number of calls")
        plt.grid(True)
        plt.ylim((0, 1.1 * costs[0]))
        plt.xlim(left=0)
        plt.xlabel("Number of calls")
        plt.ylabel("Best value")
        if filepath is not None:
            # plt.subplots(figsize=(8, 6))
            plt.savefig(filepath)
        else:
            plt.show()


@dataclass
class NelderMead(SimplexOptimizer):
    """
    Pluses:
        - no gradient -> not sensitive to noise
        - simple (no maths)
    Minuses :
        - cases of convergence toward a non-stationary point (grad != 0)
        - costly (many function calls)
    """

    alpha: float  # reflexion, 1
    beta: float  # expansion, 2
    gamma: float  # contraction/reduction, 1/2

    def _minimization_step(self, counted_function: CountedFunction, state: OptimizationState) -> str:
        # sort already happened
        simplex = state.simplex
        fsimplex = state.fsimplex
        best, fbest = simplex[0].copy(), fsimplex[0].copy()
        worst, fworst = simplex[-1].copy(), fsimplex[-1].copy()

        barycenter = simplex[:-1].mean(axis=0)  # without worst point
        reflected = (1 + self.alpha) * barycenter - self.alpha * worst
        freflected = counted_function(reflected)
        if freflected < fbest:
            expansion = (1 + self.beta) * barycenter - self.beta * worst
            fexpansion = counted_function(expansion)
            if fexpansion < freflected:
                movement = "Expansion"
                simplex[-1] = expansion
                fsimplex[-1] = fexpansion
            else:
                movement = "Reflection (freflected < fbest)"
                simplex[-1] = reflected
                fsimplex[-1] = freflected
        else:
            if freflected < fworst:
                movement = "Reflection (freflected >= fbest)"
                simplex[-1] = reflected
                fsimplex[-1] = freflected
            else:
                contracted = (1 - self.gamma) * barycenter + self.gamma * worst
                fcontracted = counted_function(contracted)
                if fcontracted < fworst:
                    movement = "Contraction"
                    simplex[-1] = contracted
                    fsimplex[-1] = fcontracted
                else:
                    movement = "Reduction"
                    # for numba, use np.broadcast: simplex[:] = (1 - self.gamma) * np.broadcast(best, simplex.shape) + self.gamma * simplex
                    simplex[:] = (1 - self.gamma) * best + self.gamma * simplex
                    fsimplex[:] = np.array([counted_function(x) for x in simplex])

        return movement


@dataclass
class Torczon(SimplexOptimizer):
    """
    https://www.researchgate.net/publication/2674306_Direct_Search_Methods_On_Parallel_Machines
    Pluses over Nelder-Mead:
        - paralellizable
        - proof of convergence toward a local minimum
    """

    alpha: float  # reflexion, 1
    beta: float  # expansion, 2
    gamma: float  # contraction, 1/2

    def _minimization_step(self, counted_function: CountedFunction, state: OptimizationState) -> str:
        # sort already happened
        simplex = state.simplex
        fsimplex = state.fsimplex
        best, fbest = simplex[0].copy(), fsimplex[0].copy()
        reflexion = (1 + self.alpha) * best - self.alpha * simplex
        freflexion = np.array([counted_function(x) for x in reflexion])
        if any(freflexion < fbest):
            movement = "Expansion"
            simplex[:] = (1 + self.beta) * best - self.beta * simplex
        else:
            movement = "Contraction"
            simplex[:] = (1 - self.gamma) * best + self.gamma * simplex
        fsimplex[:] = np.array([counted_function(x) for x in simplex])

        return movement
