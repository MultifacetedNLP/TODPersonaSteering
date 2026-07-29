"""Unified interactive E2E-TOD inference entrypoint (SGD).

This is the canonical name for the inference entrypoint. All logic lives in
:mod:`entrypoints.inference.t5_tod_interactive_inference_modified`; this
module re-exports the Hydra entry-point so it can be invoked as:

    python -m entrypoints.inference.t5_tod_interactive_inference

The active config lives in ``config/hydra/tod_runs/tod_inference.yaml``.
"""

from entrypoints.inference.t5_tod_interactive_inference_modified import (  # noqa: F401
    hydra_start,
    T5Tod,
    _apply_env_overrides,
)

if __name__ == "__main__":
    hydra_start()
