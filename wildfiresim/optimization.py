from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, ClassVar, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from tqdm.std import tqdm

from wildfiresim.get_frames import get_frames
from wildfiresim.simulation import Simulation, SimulationState
from wildfiresim.vprint import vprint


def count_func_calls(function, kwargs):
    ncalls = [0]

    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, **kwargs)

    return ncalls, function_wrapper


OptVar = np.ndarray
CostHistory = List[Tuple[int, OptVar, float]]
CostFunction = Callable[..., float]


@dataclass
class SimplexOptimizer(ABC):
    xtol: float = 1e-3
    ftol: float = 1e-3
    maxiter: int = 400
    maxfun: int = 400
    verbose: bool = False
    history: Optional[CostHistory] = field(init=False)
    pattern: str = "{:>10s}{:>10s}{:>10s}{:>10s}{:>35s}{:>25s}{:>10s}"
    
    # abstract
    name: ClassVar[str] = field(init=False)

    def minimize(
        self, function: CostFunction, initial_simplex, funckwargs: dict = {}, track_cost: bool = False, verbose: bool = False
    ) -> Tuple[np.ndarray, float, bool]:
        """find x that minimizes function(x, **funcargs)"""
        # best x is simplex[0]
        ncalls, wrapped_function = count_func_calls(function, funckwargs)

        self._print_header(verbose=verbose)

        simplex = initial_simplex.copy()
        fsimplex = np.array([wrapped_function(x) for x in simplex])
        self._sort_simplex(simplex, fsimplex)

        nb_iter = 0
        self.history = [] if track_cost else None
        converged = False
        while ncalls[0] < self.maxfun and nb_iter < self.maxiter:
            nb_iter += 1

            movement = self._minimization_step(wrapped_function, simplex, fsimplex)
            self._sort_simplex(simplex, fsimplex)
            best, fbest = simplex[0], fsimplex[0]
            xspread, fspread = (np.abs(simplex[1:] - best).max(), np.abs(fsimplex[1:] - fbest).max())

            self._print_row(
                nb_iter,
                ncalls[0],
                (xspread := np.abs(simplex[1:] - best).max()),
                (fspread := np.abs(fsimplex[1:] - fbest).max()),
                movement,
                (best := simplex[0]),
                (fbest := fsimplex[0]),
                verbose=verbose,
            )

            self._track_cost(nb_iter, best, fbest)

            if xspread < self.xtol and fspread < self.ftol:
                converged = True
                break

        reason = (
            "xspread < self.xtol and fspread < self.ftol"
            if converged
            else "too many function calls"
            if ncalls[0] > self.maxfun
            else "too many iterations"
        )
        vprint(f"Stopping minimization at {nb_iter=} ({converged=}) because {reason}.\n", verbose=verbose)
        return simplex[0], fsimplex[0], converged

    @abstractmethod
    def _minimization_step(self, wrapped_function, simplex, fsimplex):
        """
        modify simplex and fsimplex to advance one step
        When a vertex of the simplex is updated, its corresponding cost in fsimmplex must be updated aswell
        """

    def _sort_simplex(self, simplex, fsimplex):
        # best: ind = 0
        # worst: ind = -1
        ind = np.argsort(fsimplex)
        simplex.take(ind, out=simplex, axis=0)
        fsimplex.take(ind, out=fsimplex)

    def _track_cost(self, nb_iter: int, x: OptVar, cost: float):
        if self.history is not None:
            if len(self.history) == 0 or np.any(self.history[-1][1] != x):
                self.history.append((nb_iter, deepcopy(x), cost))

    def _print_header(self, verbose):
        vprint(
            self.pattern.format("nb_iter", "nb_calls", "x_spread", "f_spread", "movement (current to next)", "best", "f(best)"),
            verbose=verbose,
            timestamp=True,
        )

    def _print_row(self, nb_iter, ncalls, xspread, fspread, movement, best, fbest, verbose):
        rounding = 2
        vprint(
            self.pattern.format(
                *map(
                    str,
                    (
                        nb_iter,
                        ncalls,
                        round(xspread, rounding),
                        round(fspread, rounding),
                        movement,
                        np.round(best, rounding),
                        round(fbest, rounding),
                    ),
                )
            ),
            verbose=verbose,
            timestamp=True,
        )

    def animate(
        self, initial_state: SimulationState, simulation: Simulation, nb_frames: int = None, title: str = "", show_final_state: bool = False
    ) -> FuncAnimation:
        """
        If show_final_state == True, display the final distribution instead (every simulation needs to be recomputed)
        """

        if self.history is None:
            raise Exception("Tracking was disabled during the previous call to `simulate`. Retry after activating tracking.")

        if nb_frames is None:
            nb_frames = len(self.history)

        if show_final_state:
            final_states = [simulation.simulate(initial_state, x) for _, x, _ in tqdm(self.history, desc="Running simulations...")]

        X, Y = simulation.forest.X, simulation.forest.Y

        fig, ax = plt.subplots()

        state = final_states[0] if show_final_state else initial_state  # type: ignore
        fuel = ax.pcolormesh(X, Y, state.c, cmap=plt.cm.YlGn, vmin=0, vmax=10)  # type: ignore
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
        ax_title = ax.text(0.5, 0.05, "", bbox={"facecolor": "w", "alpha": 0.5, "pad": 5}, transform=ax.transAxes, ha="center")

        def init():
            fuel.set_array(np.ma.array(X, mask=True))
            firewall.set_bounds(0, 0, 0, 0)
            ax.add_patch(firewall)
            ax_title.set_text("")
            return fuel, firewall, wind, ax_title

        def animate(frame: int, *_):
            k, x, fx = self.history[frame]  # type: ignore
            state = final_states[frame] if show_final_state else initial_state  # type: ignore
            fuel.set_array(state.c.ravel())
            xmin, xmax, ymin, ymax = x
            firewall.set_bounds(xmin, ymin, xmax - xmin, ymax - ymin)
            title_ = title + "\nstep={step}/{tot_step}, cost={cost:.2f}\nx={x}".format(step=k, tot_step=self.history[-1][0], cost=fx, x=np.round(x, 2))  # type: ignore
            ax_title.set_text(title_)
            return fuel, firewall, wind, ax_title

        frames = get_frames(len(self.history), nb_frames)
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.history), interval=1000, repeat=True, repeat_delay=2000, blit=True)  # type: ignore

        plt.show()

        return anim


@dataclass
class NelderMead(SimplexOptimizer):
    """
    Pluses:
        - no gradient -> not sennsitive to noise
        - simple (no maths)
    Minuses :
        - cases of convergence toward a non-stationary point (grad != 0)
        - costly (many function calls)
    """

    alpha: float = 1
    beta: float = 2
    gamma: float = 1 / 2

    def _minimization_step(self, wrapped_function, simplex, fsimplex):
        # sort already happened
        best, fbest = simplex[0].copy(), fsimplex[0].copy()
        worst, fworst = simplex[-1].copy(), fsimplex[-1].copy()

        barycenter = simplex[:-1].mean(axis=0)  # without worst point
        reflected = (1 + self.alpha) * barycenter - self.alpha * worst
        freflected = wrapped_function(reflected)
        if freflected < fbest:
            expansion = (1 + self.beta) * barycenter - self.beta * worst
            fexpansion = wrapped_function(expansion)
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
                fcontracted = wrapped_function(contracted)
                if fcontracted < fworst:
                    movement = "Contraction"
                    simplex[-1] = contracted
                    fsimplex[-1] = fcontracted
                else:
                    movement = "Reduction"
                    # simplex[:] = (1 - self.gamma) * np.broadcast(best, simplex.shape) + self.gamma * simplex
                    simplex[:] = (1 - self.gamma) * best + self.gamma * simplex
                    fsimplex[:] = np.array([wrapped_function(x) for x in simplex])

        return movement


@dataclass
class Torczon(SimplexOptimizer):
    """
    Pluses over Nelder-Mead:
        - paralellizable
        - proof of convergence toward a local minimum
    """

    alpha: float = 1 / 2
    beta: float = 1 / 2
    gamma: float = 2

    def _minimization_step(self, wrapped_function, simplex, fsimplex):
        # sort already happened
        best, fbest = simplex[0].copy(), fsimplex[0].copy()
        reflexion = (1 + self.alpha) * best - self.alpha * simplex
        freflexion = np.array([wrapped_function(x) for x in reflexion])
        if any(freflexion < fbest):
            movement = "Expansion"
            simplex[:] = (1 + self.gamma) * best - self.gamma * simplex
        else:
            movement = "Contraction"
            simplex[:] = (1 - self.beta) * best + self.beta * simplex
        fsimplex[:] = np.array([wrapped_function(x) for x in simplex])

        return movement
