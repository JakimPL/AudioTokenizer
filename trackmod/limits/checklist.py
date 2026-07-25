from trackmod.limits.capability import Capability
from trackmod.limits.table import Limits
from trackmod.limits.violation import Violation


class Checklist:
    """Accumulates the violations of a whole module so a caller sees every problem in one report."""

    def __init__(self, limits: Limits) -> None:
        self._limits = limits
        self._violations: list[Violation] = []

    def check(self, capability: Capability, value: int, *, subject: str) -> None:
        """Grade ``value`` and record it when it violates the format."""
        violation = self._limits.check(capability, value, subject=subject)
        if violation is not None:
            self._violations.append(violation)

    def report(self, violation: Violation) -> None:
        """Record a violation a caller graded itself."""
        self._violations.append(violation)

    @property
    def violations(self) -> tuple[Violation, ...]:
        """Everything recorded so far, in the order it was found."""
        return tuple(self._violations)
