"""The knobs that drive one compilation, validated once at the boundary and threaded down.

A configuration fixes the output format, the timing (``speed``/``tempo``, which derive the block length),
the dictionary size, the sparsity floor and the storage depth, plus the compliance level every bound is
read at. Fields the caller varies per run carry no defaults; the few genuinely seldom-changed settings
default to a named constant.

Compliance is what used to be a "profile", and it now means something the format itself states rather than
a pair of numbers repeated here. :attr:`~trackmod.limits.compliance.Compliance.CANONICAL` holds the module
to what the tracker it names actually honoured — 64 channels for Impulse Tracker, 32 for FastTracker 2.
:attr:`~trackmod.limits.compliance.Compliance.EXTENDED` holds it to what the record layout can physically
store, which is where the wide modules live: 127 IT channels, 192 XM channels, and a 16-bit XM tempo that
reaches rows an 8-bit one cannot. Impulse Tracker's tempo is a single byte at both levels, so asking for a
16-bit tempo there is refused rather than silently written into a field too small for it.

Two rate levers trade a little fidelity for pattern bytes, both off by default so the codec is unchanged
until asked: ``persistence`` penalises switching a channel to a new atom (a saved note byte), and
``polarity_sticky`` keeps a channel's sign through a sign flip whose volume code is that small (the tiny
opposite-sign cells that cost a note byte for almost no energy).

``taper_alpha`` is a quality default, not a rate lever: a short cosine taper on each atom's edges forces
it to start and end at silence, so a retriggered one-shot sample cannot click. ``0`` reproduces the raw,
clicking output for A/B; the default is a small fraction that removes the click for a shallow, tunable
row-rate tremolo.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from trackmod.limits.capability import Capability
from trackmod.limits.compliance import Compliance
from trackmod.limits.table import Limits
from trackmod.spec.clock import MIN_SPEED, MIN_TEMPO
from trackmod.spec.levels import MAX_VOLUME

from audiotokenizer.module.catalog import binding_for
from audiotokenizer.module.format import Format
from audiotokenizer.module.samples import POLARITIES

DEFAULT_SPEED: Final = 1
DEFAULT_MIN_ENERGY: Final = 0.0
DEFAULT_PERSISTENCE: Final = 0.0  # λ; 0 keeps selection independent per row
DEFAULT_POLARITY_STICKY: Final = 0  # volume-code threshold; 0 never suppresses a sign flip
DEFAULT_TAPER_ALPHA: Final = 0.2  # Tukey fraction that zeros atom edges; 0 disables (raw, clicking)
DEFAULT_PCM_BITS: Final = 8
DEFAULT_BUDGET_BYTES: Final = 2 * 1024 * 1024
DEFAULT_NAME: Final = "audiotokenizer"
DEFAULT_FORMAT: Final = Format.IT
DEFAULT_COMPLIANCE: Final = Compliance.EXTENDED


def max_atoms_for(module_format: Format, compliance: Compliance) -> int:
    """The widest dictionary a format can route, each atom costing the two samples its polarities need."""
    limits = binding_for(module_format).limits(compliance)
    return limits.bound(Capability.SAMPLES).maximum // POLARITIES


#: The widest pool any format admits, which is the field bound; :attr:`TokenizerConfig.max_atoms` is the
#: per-format cap the validator enforces.
WIDEST_ATOMS: Final = max(max_atoms_for(module_format, Compliance.EXTENDED) for module_format in Format)


class ConfigModel(BaseModel):
    """Frozen, strict base: immutable, hashable, and rejecting unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TokenizerConfig(ConfigModel):
    """One end-to-end compilation's settings, checked against what the chosen format can carry."""

    compliance: Compliance
    tempo: int = Field(ge=MIN_TEMPO)
    n_atoms: int = Field(ge=1, le=WIDEST_ATOMS)
    format: Format = DEFAULT_FORMAT
    speed: int = Field(default=DEFAULT_SPEED, ge=MIN_SPEED)
    min_energy: float = Field(default=DEFAULT_MIN_ENERGY, ge=0.0)
    persistence: float = Field(default=DEFAULT_PERSISTENCE, ge=0.0)
    polarity_sticky: int = Field(default=DEFAULT_POLARITY_STICKY, ge=0, le=MAX_VOLUME)
    taper_alpha: float = Field(default=DEFAULT_TAPER_ALPHA, ge=0.0, le=1.0)
    pcm_bits: Literal[8, 16] = DEFAULT_PCM_BITS
    budget_bytes: int = Field(default=DEFAULT_BUDGET_BYTES, ge=1)
    name: str = DEFAULT_NAME

    @property
    def limits(self) -> Limits:
        """The bounds the chosen format holds this compilation to."""
        return binding_for(self.format).limits(self.compliance)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_channels(self) -> int:
        """The channel cap the format allows at this compliance level."""
        return self.limits.bound(Capability.CHANNELS).maximum

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_tempo(self) -> int:
        """The tempo ceiling the format allows at this compliance level."""
        return self.limits.bound(Capability.TEMPO).maximum

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_atoms(self) -> int:
        """The dictionary pool the format can route, two stored samples to an atom."""
        return max_atoms_for(self.format, self.compliance)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_per_row(self) -> int:
        """Atoms kept per block: the pool, capped by the channels the compliance level permits."""
        return min(self.n_atoms, self.max_channels)

    @model_validator(mode="after")
    def _within_the_format(self) -> TokenizerConfig:
        """Hold every quantity the format bounds to what it can carry at this compliance level.

        Raises:
            ValueError: when the pool needs more stored samples than the format routes, or when the
                timing leaves the speed or tempo range the format reads.
        """
        if self.n_atoms > self.max_atoms:
            raise ValueError(f"n_atoms {self.n_atoms} exceeds the {self.format} routing limit of {self.max_atoms}")

        for capability, value in ((Capability.TEMPO, self.tempo), (Capability.SPEED, self.speed)):
            bound = self.limits.bound(capability)
            if not bound.contains(value):
                raise ValueError(f"{capability} {value} is outside what {self.format} reads at {self.compliance}")

        return self
