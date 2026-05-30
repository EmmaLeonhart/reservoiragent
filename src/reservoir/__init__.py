"""reservoir — the Reservoir Agent research package.

A fixed, randomly-initialized echo-state reservoir that can be driven by, and
injected into, a pretrained transformer's mid-layer attention so the model carries
state across forward passes. This package holds the reservoir core, dynamics
metrics, the spectral-radius sweep, and the model-surgery hooks.

See `FINDINGS.md` for the write-up and `literature/REVIEW.md` for the grounding.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
