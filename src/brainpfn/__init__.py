"""BrainPFN: in-context prediction of behaviour from resting-state functional connectivity."""

from importlib.metadata import version

from brainpfn.connectivity import fc_to_logchol
from brainpfn.estimator import BrainPFN
from brainpfn.hub import DEFAULT_VARIANT, VARIANTS

__version__ = version("brainpfn")

__all__ = ["BrainPFN", "fc_to_logchol", "VARIANTS", "DEFAULT_VARIANT", "__version__"]
