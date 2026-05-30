"""Echo-state reservoir core.

A fixed, randomly-initialized reservoir in the classical reservoir-computing sense
(Jaeger 2001): the recurrent weights ``W_r`` and input weights ``W_in`` are random
and never trained; only a downstream readout (added later) is learned. The reservoir
state evolves by the leaky echo-state update

    r(t) = (1 - a) * r(t-1) + a * tanh( W_r @ r(t-1) + W_in @ x(t) )

with ``a`` the leak rate. ``W_r`` is sparse and rescaled to a target **spectral
radius** — the knob that places the reservoir on the contraction ↔ edge-of-chaos
spectrum the dynamics study characterizes. See ``literature/REVIEW.md`` for why the
spectral radius (and the echo state property) govern usable-signal-vs-noise.
"""
from __future__ import annotations

import numpy as np


class EchoStateReservoir:
    """A fixed random echo-state reservoir.

    Parameters
    ----------
    n_reservoir : int
        Number of reservoir units, K.
    n_input : int
        Dimension of the input driving the reservoir each step.
    spectral_radius : float
        Target spectral radius of ``W_r`` (it is rescaled to exactly this).
    input_scaling : float
        Scale of the uniform ``W_in`` entries.
    sparsity : float
        Fraction of ``W_r`` entries that are nonzero (in (0, 1]).
    leak_rate : float
        Leaking rate ``a`` in [0, 1]; 1.0 = no leak, 0.0 = frozen state.
    seed : int
        RNG seed; fixes both ``W_in`` and ``W_r`` and thus the whole reservoir.
    dtype : np.dtype
        State / weight dtype (default float64).
    """

    def __init__(self, n_reservoir: int, n_input: int, *, spectral_radius: float = 0.9,
                 input_scaling: float = 1.0, sparsity: float = 0.1, leak_rate: float = 1.0,
                 seed: int = 0, dtype=np.float64):
        if not (0.0 < sparsity <= 1.0):
            raise ValueError("sparsity must be in (0, 1]")
        if not (0.0 <= leak_rate <= 1.0):
            raise ValueError("leak_rate must be in [0, 1]")
        self.n_reservoir = int(n_reservoir)
        self.n_input = int(n_input)
        self.spectral_radius = float(spectral_radius)
        self.input_scaling = float(input_scaling)
        self.sparsity = float(sparsity)
        self.leak_rate = float(leak_rate)
        self.seed = int(seed)
        self.dtype = dtype

        rng = np.random.default_rng(self.seed)
        # input weights: dense uniform[-1, 1] * input_scaling
        self.W_in = (rng.uniform(-1.0, 1.0, size=(self.n_reservoir, self.n_input))
                     * self.input_scaling).astype(dtype)
        # recurrent weights: sparse uniform[-1, 1], then rescale to target radius
        mask = rng.random((self.n_reservoir, self.n_reservoir)) < self.sparsity
        W = rng.uniform(-1.0, 1.0, size=(self.n_reservoir, self.n_reservoir)) * mask
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 0:
            W = W * (self.spectral_radius / rho)
        self.W_r = W.astype(dtype)

        self.state = np.zeros(self.n_reservoir, dtype=dtype)

    def spectral_radius_actual(self) -> float:
        """The realised spectral radius of ``W_r`` (max |eigenvalue|)."""
        return float(np.max(np.abs(np.linalg.eigvals(self.W_r))))

    def reset(self, state=None) -> None:
        """Reset the reservoir state (to zeros, or to a provided vector)."""
        if state is None:
            self.state = np.zeros(self.n_reservoir, dtype=self.dtype)
        else:
            state = np.asarray(state, dtype=self.dtype)
            if state.shape != (self.n_reservoir,):
                raise ValueError(f"state must have shape ({self.n_reservoir},)")
            self.state = state.copy()

    def step(self, x) -> np.ndarray:
        """Advance one tick with input ``x`` (shape ``(n_input,)``); return the state."""
        x = np.asarray(x, dtype=self.dtype)
        pre = self.W_r @ self.state + self.W_in @ x
        self.state = (1.0 - self.leak_rate) * self.state + self.leak_rate * np.tanh(pre)
        return self.state

    def run(self, inputs, *, reset: bool = True) -> np.ndarray:
        """Drive the reservoir with ``inputs`` (shape ``(T, n_input)``).

        Returns the state trajectory, shape ``(T, n_reservoir)``. If ``reset`` is
        True the state is zeroed first; otherwise the current state is the start.
        """
        inputs = np.asarray(inputs, dtype=self.dtype)
        if inputs.ndim != 2 or inputs.shape[1] != self.n_input:
            raise ValueError(f"inputs must have shape (T, {self.n_input})")
        if reset:
            self.reset()
        out = np.empty((inputs.shape[0], self.n_reservoir), dtype=self.dtype)
        for t in range(inputs.shape[0]):
            out[t] = self.step(inputs[t])
        return out
