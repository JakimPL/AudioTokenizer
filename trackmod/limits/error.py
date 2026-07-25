from collections.abc import Sequence

from trackmod.limits.violation import Violation


class LimitError(ValueError):
    """Raised when a module carries values a format refuses to store at its compliance level."""

    def __init__(self, violations: Sequence[Violation]) -> None:
        self.violations = tuple(violations)
        super().__init__("; ".join(str(violation) for violation in self.violations))


def require(violations: Sequence[Violation]) -> None:
    """Raise :class:`LimitError` when any violation was collected.

    Raises:
        LimitError: when ``violations`` is non-empty.
    """
    if violations:
        raise LimitError(violations)
