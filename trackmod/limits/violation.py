from pydantic import BaseModel

from trackmod.limits.bound import Bound
from trackmod.limits.capability import Capability
from trackmod.limits.severity import Severity
from trackmod.schema.config import FROZEN


class Violation(BaseModel):
    """One value that falls outside a format's bound, with the subject that carried it."""

    model_config = FROZEN

    capability: Capability
    value: int
    bound: Bound
    severity: Severity
    subject: str

    def __str__(self) -> str:
        return f"{self.subject}: {self.capability} is {self.value}, outside {self.bound} ({self.severity})"
