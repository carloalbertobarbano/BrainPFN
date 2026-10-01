"""BrainPFN: in-context prediction of behaviour from resting-state functional connectivity."""

from brainpfn.connectivity import fc_to_logchol
from brainpfn.estimator import BrainPFN
from brainpfn.hub import DEFAULT_VARIANT, VARIANTS

__version__ = "0.1.0"

__all__ = ["BrainPFN", "fc_to_logchol", "VARIANTS", "DEFAULT_VARIANT", "__version__"]
