"""The knobs that drive one compilation, validated once at the boundary and threaded down.

A configuration fixes the timing (``speed``/``tempo``, which derive the block length), the dictionary
size, the sparsity floor and the storage depth, plus the profile that caps the channel count — 64 for a
canonical Impulse Tracker file, 127 for the OpenMPT-only "hacked" reach. Fields the caller varies per run
carry no defaults; the few genuinely seldom-changed settings default to a named constant.

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

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from audiotokenizer.it.spec import (
    HACKED_MAX_CHANNELS,
    HACKED_MAX_TEMPO,
    MAX_SAMPLES,
    MAX_SPEED,
    MAX_TEMPO,
    MAX_VOLUME,
    MIN_SPEED,
    MIN_TEMPO,
    STRICT_MAX_CHANNELS,
    STRICT_MAX_TEMPO,
)
from audiotokenizer.xm.spec import HACKED_MAX_CHANNELS as XM_HACKED_MAX_CHANNELS
from audiotokenizer.xm.spec import HACKED_MAX_TEMPO as XM_HACKED_MAX_TEMPO
from audiotokenizer.xm.spec import MAX_SAMPLES_TOTAL as XM_MAX_SAMPLES_TOTAL
from audiotokenizer.xm.spec import STRICT_MAX_CHANNELS as XM_STRICT_MAX_CHANNELS
from audiotokenizer.xm.spec import STRICT_MAX_TEMPO as XM_STRICT_MAX_TEMPO

Profile = Literal["strict", "hacked"]
Format = Literal["it", "xm"]

DEFAULT_SPEED: Final = 1
DEFAULT_MIN_ENERGY: Final = 0.0
DEFAULT_PERSISTENCE: Final = 0.0  # λ; 0 keeps selection independent per row
DEFAULT_POLARITY_STICKY: Final = 0  # volume-code threshold; 0 never suppresses a sign flip
DEFAULT_TAPER_ALPHA: Final = 0.2  # Tukey fraction that zeros atom edges; 0 disables (raw, clicking)
DEFAULT_PCM_BITS: Final = 8
DEFAULT_BUDGET_BYTES: Final = 2 * 1024 * 1024
DEFAULT_NAME: Final = "audiotokenizer"
DEFAULT_FORMAT: Final[Format] = "it"
#: Each atom stores two samples. IT's 1-byte note map routes at most 255 samples -> 127 atoms; XM routes up
#: to 128 instruments x 16 samples -> 1024 atoms. ``MAX_ATOMS`` is the IT ceiling (the planner is IT-only);
#: the per-format cap is :attr:`TokenizerConfig.max_atoms`.
MAX_ATOMS: Final = MAX_SAMPLES // 2  # 127
XM_MAX_ATOMS: Final = XM_MAX_SAMPLES_TOTAL // 2  # 1024, the widest pool any format admits


class ConfigModel(BaseModel):
    """Frozen, strict base: immutable, hashable, and rejecting unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TokenizerConfig(ConfigModel):
    """One end-to-end compilation's settings."""

    profile: Profile
    tempo: int = Field(ge=MIN_TEMPO, le=MAX_TEMPO)
    n_atoms: int = Field(ge=1, le=XM_MAX_ATOMS)  # the widest any format admits; :attr:`max_atoms` caps per format
    format: Format = DEFAULT_FORMAT
    speed: int = Field(default=DEFAULT_SPEED, ge=MIN_SPEED, le=MAX_SPEED)
    min_energy: float = Field(default=DEFAULT_MIN_ENERGY, ge=0.0)
    persistence: float = Field(default=DEFAULT_PERSISTENCE, ge=0.0)
    polarity_sticky: int = Field(default=DEFAULT_POLARITY_STICKY, ge=0, le=MAX_VOLUME)
    taper_alpha: float = Field(default=DEFAULT_TAPER_ALPHA, ge=0.0, le=1.0)
    pcm_bits: Literal[8, 16] = DEFAULT_PCM_BITS
    budget_bytes: int = Field(default=DEFAULT_BUDGET_BYTES, ge=1)
    name: str = DEFAULT_NAME

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hacked(self) -> bool:
        return self.profile == "hacked"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_channels(self) -> int:
        """The channel cap the format and profile allow: IT 64/127, XM 32/255 (strict/hacked)."""
        if self.format == "xm":
            return XM_HACKED_MAX_CHANNELS if self.hacked else XM_STRICT_MAX_CHANNELS
        return HACKED_MAX_CHANNELS if self.hacked else STRICT_MAX_CHANNELS

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_tempo(self) -> int:
        """The tempo ceiling: 255 for strict, the 16-bit hack (65535) for hacked — same in both formats."""
        if self.format == "xm":
            return XM_HACKED_MAX_TEMPO if self.hacked else XM_STRICT_MAX_TEMPO
        return HACKED_MAX_TEMPO if self.hacked else STRICT_MAX_TEMPO

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_atoms(self) -> int:
        """The dictionary-pool cap the format can route: IT 127 samples-pairs, XM up to 1024."""
        return XM_MAX_ATOMS if self.format == "xm" else MAX_ATOMS

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_per_row(self) -> int:
        """Atoms kept per block: the pool, capped by the channels the profile permits."""
        return min(self.n_atoms, self.max_channels)

    @model_validator(mode="after")
    def _atoms_fit_the_format(self) -> TokenizerConfig:
        """Reject a pool the format cannot route (skipped on the hacked ``model_construct`` path)."""
        if self.n_atoms > self.max_atoms:
            raise ValueError(f"n_atoms {self.n_atoms} exceeds the {self.format} routing limit of {self.max_atoms}")
        return self

    @classmethod
    def load(cls, *, profile: Profile, **arguments: Any) -> TokenizerConfig:
        if profile == "hacked":
            return TokenizerConfig.model_construct(profile=profile, **arguments)

        return TokenizerConfig(profile=profile, **arguments)
