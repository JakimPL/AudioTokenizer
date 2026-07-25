from trackmod.limits.compliance import Compliance
from trackmod.limits.table import Limits
from trackmod.xm.spec.capacities import CAPACITIES


def xm_limits(compliance: Compliance) -> Limits:
    """The bounds a FastTracker 2 module is held to at one compliance level."""
    return Limits(compliance=compliance, capacities=CAPACITIES)
