from __future__ import annotations

import pytest

from trackmod.limits.bound import Bound
from trackmod.limits.capability import Capability
from trackmod.limits.capacity import Capacity
from trackmod.limits.checklist import Checklist
from trackmod.limits.compliance import Compliance
from trackmod.limits.error import LimitError, require
from trackmod.limits.guard import require_range
from trackmod.limits.severity import Severity
from trackmod.limits.table import Limits

CAPACITIES = {
    Capability.TEMPO: Capacity(
        canonical=Bound(minimum=32, maximum=255),
        structural=Bound(minimum=32, maximum=65535),
    ),
    Capability.CHANNELS: Capacity.fixed(Bound(minimum=1, maximum=64)),
}


def limits(compliance: Compliance) -> Limits:
    return Limits(compliance=compliance, capacities=CAPACITIES)


@pytest.mark.parametrize("compliance", list(Compliance))
def test_a_value_inside_the_canonical_bound_passes_at_every_level(compliance: Compliance) -> None:
    assert limits(compliance).check(Capability.TEMPO, 125, subject="song") is None


def test_headroom_is_a_compliance_violation_only_for_the_original_tracker() -> None:
    canonical = limits(Compliance.CANONICAL).check(Capability.TEMPO, 441, subject="song")
    assert canonical is not None
    assert canonical.severity is Severity.COMPLIANCE
    assert limits(Compliance.EXTENDED).check(Capability.TEMPO, 441, subject="song") is None


@pytest.mark.parametrize("compliance", list(Compliance))
def test_a_value_the_field_cannot_hold_always_fails(compliance: Compliance) -> None:
    violation = limits(compliance).check(Capability.TEMPO, 70000, subject="song")
    assert violation is not None
    assert violation.severity is Severity.STRUCTURAL


def test_a_capability_with_no_headroom_fails_identically_at_both_levels() -> None:
    severities = {limits(compliance).check(Capability.CHANNELS, 65, subject="song") for compliance in Compliance}
    assert all(violation is not None and violation.severity is Severity.STRUCTURAL for violation in severities)


def test_the_effective_bound_widens_with_compliance() -> None:
    assert limits(Compliance.CANONICAL).bound(Capability.TEMPO).maximum == 255
    assert limits(Compliance.EXTENDED).bound(Capability.TEMPO).maximum == 65535


def test_a_canonical_bound_outside_the_structural_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        Capacity(canonical=Bound(minimum=32, maximum=512), structural=Bound(minimum=32, maximum=255))


def test_an_empty_bound_is_rejected() -> None:
    with pytest.raises(ValueError):
        Bound(minimum=10, maximum=9)


def test_a_checklist_collects_every_problem_before_raising() -> None:
    checklist = Checklist(limits(Compliance.CANONICAL))
    checklist.check(Capability.TEMPO, 441, subject="song")
    checklist.check(Capability.CHANNELS, 128, subject="song")
    checklist.check(Capability.TEMPO, 125, subject="song")
    assert len(checklist.violations) == 2
    with pytest.raises(LimitError) as error:
        require(checklist.violations)

    assert error.value.violations == checklist.violations


def test_an_empty_checklist_raises_nothing() -> None:
    require(Checklist(limits(Compliance.CANONICAL)).violations)


def test_a_guarded_value_inside_its_bound_passes_through() -> None:
    bound = Bound(minimum=0, maximum=15)
    assert require_range(15, bound=bound, subject="delay") == 15
    with pytest.raises(ValueError):
        require_range(16, bound=bound, subject="delay")
