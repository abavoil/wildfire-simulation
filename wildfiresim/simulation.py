from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Union

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from wildfiresim.get_frames import get_frames
from wildfiresim.no_history_exception import NoHistoryException
from wildfiresim.vprint import vprint

mpl.rcParams["pcolor.shading"] = "nearest"


def disk(X, Y, x, y, r):
    return (X - x) ** 2 + (Y - y) ** 2 <= r ** 2


def rect(X, Y, xmin, xmax, ymin, ymax):
    return np.logical_and.reduce((xmin <= X, X <= xmax, ymin <= Y, Y <= ymax))


@dataclass
class Forest:
    N: int
    T_fire: float
    mu: float
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)
    u: np.ndarray = field(init=False, repr=False)
    v: np.ndarray = field(init=False, repr=False)
    dx: float = field(init=False)
    dt: float = field(init=False)

    def __post_init__(self):
        x, self.dx = np.linspace(0, 1, self.N, retstep=True)
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")
        self.u = np.cos(np.pi * self.Y)
        self.v = 0.6 * np.sin(np.pi / 2 * (self.X + 0.2))

        dt_c = self.dx / np.sqrt(self.u ** 2 + self.v ** 2).max()
        dt_d = self.dx ** 2 / (2 * self.mu)
        self.dt = min(dt_c, dt_d) / 4


@dataclass
class SimulationState:
    t: float
    T: np.ndarray = field(repr=False)
    c: np.ndarray = field(repr=False)

    def cut_trees(self, X: np.ndarray, Y: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float):
        self.c[rect(X, Y, xmin, xmax, ymin, ymax)] = -1

    def get_masked_T(self, T_fire: float) -> np.ma.MaskedArray:
        return np.ma.masked_where(self.T < T_fire, self.T)

    def get_fuel_amount(self, dx: float):
        return self.c.sum() * dx * dx

    @staticmethod
    def create_initial_state(
        X: np.ndarray,
        Y: np.ndarray,
        x0: float,
        y0: float,
        r0: float,
        T_init_fire: float,
        c_init: float,
        n_circles: int = 0,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> SimulationState:
        """
        rng: either None for non-reproducibility, int for seed, or Generator
        """
        T = np.zeros(X.shape)
        T[disk(X, Y, x0, y0, r0)] = T_init_fire
        c = np.full(X.shape, c_init, dtype=float)

        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        for _ in range(n_circles):
            rr = rng.uniform(0.1, 0.2)
            xr = rng.uniform(0.1, 0.9)
            yr = rng.uniform(0.1, 0.9)
            valr = rng.uniform(-5, 5)
            c[disk(X, Y, xr, yr, rr)] += valr

        return SimulationState(0, T, c)


@dataclass
class Simulation:
    """create simulation, simulate, get results"""

    forest: Forest
    tf: float
    max_iter: int
    history: Optional[List[SimulationState]] = field(init=False)

    def simulate(
        self, initial_state: SimulationState, x: Optional[np.ndarray] = None, track_state: bool = False, verbose: bool = False
    ) -> SimulationState:
        state = deepcopy(initial_state)

        if x is not None:
            state.cut_trees(self.forest.X, self.forest.Y, *x)

        self.history = [] if track_state else None
        self._track_state(state)
        for k, t in enumerate(np.arange(self.forest.dt, self.tf + self.forest.dt, self.forest.dt)):
            # pas de simulation
            self._simulation_step(state)

            # critères d'arrêt
            stop_msg = None
            if self.max_iter > 0 and k > self.max_iter:
                stop_msg = "Maximum number of iterations has been exceeded."
            elif state.T.max() < self.forest.T_fire:
                stop_msg = f"T < T_fire everywhere."

            # ajout à l'historique si le tracking est activé
            self._track_state(state)

            # arrêt
            if stop_msg is not None:
                vprint(f"Stopping simulation at {k=} ({t=:.2f}) because", stop_msg, verbose=verbose)
                break

        return state

    def _simulation_step(self, state: SimulationState):
        # raccourcis
        back = slice(None, -2)
        mid = slice(1, -1)
        front = slice(2, None)
        T = state.T
        c = state.c
        T_mid = T[mid, mid]
        u_mid = self.forest.u[mid, mid]
        v_mid = self.forest.v[mid, mid]
        mu = self.forest.mu
        dx = self.forest.dx
        dt = self.forest.dt
        on_fire = T > self.forest.T_fire

        # calcul des dérivées
        Tx_back = (T_mid - T[back, mid]) / dx
        Tx_front = (T[front, mid] - T_mid) / dx
        Ty_back = (T_mid - T[mid, back]) / dx
        Ty_front = (T[mid, front] - T_mid) / dx
        Txx = (Tx_front - Tx_back) / dx
        Tyy = (Ty_front - Ty_back) / dx

        # laplacien, advection, reaction
        diffusion = mu * (Txx + Tyy)

        Tx_upwind = np.where(u_mid > 0, Tx_back, Tx_front)
        Ty_upwind = np.where(v_mid > 0, Ty_back, Ty_front)
        advection = -(Tx_upwind * u_mid + Ty_upwind * v_mid)

        reaction_T = np.zeros_like(T)
        reaction_T[np.logical_and(on_fire, c >= 0)] = 10
        reaction_T[np.logical_and(on_fire, c < 0)] = -5

        # mise à jour
        state.t += dt
        T_mid += dt * ((diffusion + advection) + (reaction_T * T)[mid, mid])
        c[on_fire] += dt * -100

        # condition de neumann au bord
        T[:, 0] = T[:, 1]
        T[:, -1] = T[:, -2]
        T[0, :] = T[1, :]
        T[-1, :] = T[-2, :]

    def _track_state(self, state: SimulationState):
        if self.history is not None:
            self.history.append(deepcopy(state))

    def animate(self, nb_frames: Optional[int] = 100, title: str = "") -> FuncAnimation:
        if self.history is None:
            raise NoHistoryException()

        if nb_frames is None:
            nb_frames = len(self.history)

        X, Y = self.forest.X, self.forest.Y
        fig, ax = plt.subplots()

        fuel = ax.pcolormesh(X, Y, np.zeros_like(X), cmap=plt.cm.YlGn, vmin=0, vmax=10)  # type: ignore
        fuel_cb = fig.colorbar(fuel, ax=ax)
        fuel_cb.set_label("fuel", loc="top")

        fire = ax.pcolormesh(X, Y, np.zeros_like(X), cmap=plt.cm.hot, vmin=0, vmax=0.15)  # type: ignore
        fire_cb = fig.colorbar(fire, ax=ax)
        fire_cb.set_label("fire", loc="top")

        n_arrows = 6
        wind_ind = np.zeros(X.shape, dtype=bool)
        inds = slice((X.shape[0] % n_arrows) // 2, None, X.shape[0] // n_arrows)
        wind_ind[inds, inds] = True
        wind = ax.quiver(X[wind_ind], Y[wind_ind], self.forest.u[wind_ind], self.forest.v[wind_ind])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("")
        ax.set_title(title)
        ax_text_info = ax.text(0.5, 0.94, "", bbox={"facecolor": "w", "alpha": 0.5, "pad": 5}, transform=ax.transAxes, ha="center")

        def init():
            fuel.set_array(np.array([]))
            fire.set_array(np.array([]))
            return fuel, fire, wind, ax_text_info

        def animate(frame: int, *_):
            state = self.history[frame]  # type: ignore
            pct = int(100 * (frame + 1) / len(self.history))  # type: ignore
            ax_text_info.set_text("t = {t:.2f} s, {pct}%".format(t=state.t, pct=pct))
            fuel.set_array(state.c.ravel())
            fire.set_array(state.get_masked_T(self.forest.T_fire).ravel())
            return fuel, fire, wind, ax_text_info

        frames = get_frames(len(self.history), nb_frames)
        anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=17, repeat=True, repeat_delay=1000, blit=True)  # type: ignore

        plt.show()

        return anim
