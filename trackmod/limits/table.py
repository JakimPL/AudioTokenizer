from collections.abc import Mapping

from pydantic import BaseModel

from trackmod.limits.bound import Bound
from trackmod.limits.capability import Capability
from trackmod.limits.capacity import Capacity
from trackmod.limits.compliance import Compliance
from trackmod.limits.severity import Severity
from trackmod.limits.violation import Violation
from trackmod.schema.config import FROZEN


class Limits(BaseModel):
    """A format's capacity table read at one compliance level.

    :meth:`bound` answers what a caller may use; :meth:`check` grades a value it already has. A value the
    record layout cannot hold is always a structural violation; a value the layout holds but the original
    tracker ignores is reported only under canonical compliance, which is what lets an extended module
    carry a 16-bit tempo the format's own tracker would never read.
    """

    model_config = FROZEN

    compliance: Compliance
    capacities: Mapping[Capability, Capacity]

    def capacity(self, capability: Capability) -> Capacity:
        """The declared capacity for ``capability``.

        Raises:
            KeyError: when the format declares no capacity for ``capability``.
        """
        return self.capacities[capability]

    def bound(self, capability: Capability) -> Bound:
        """The effective bound at this compliance level."""
        capacity = self.capacity(capability)
        match self.compliance:
            case Compliance.CANONICAL:
                return capacity.canonical
            case Compliance.EXTENDED:
                return capacity.structural

    def check(self, capability: Capability, value: int, *, subject: str) -> Violation | None:
        """Grade ``value`` against ``capability``, returning the violation it commits, if any."""
        capacity = self.capacity(capability)
        if not capacity.structural.contains(value):
            return Violation(
                capability=capability,
                value=value,
                bound=capacity.structural,
                severity=Severity.STRUCTURAL,
                subject=subject,
            )

        if self.compliance is Compliance.CANONICAL and not capacity.canonical.contains(value):
            return Violation(
                capability=capability,
                value=value,
                bound=capacity.canonical,
                severity=Severity.COMPLIANCE,
                subject=subject,
            )

        return None
