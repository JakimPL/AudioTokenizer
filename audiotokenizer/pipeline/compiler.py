"""Compile a signal into a tracker module end to end, and report exactly what it cost.

The pipeline runs one pass: frame the signal, learn a dictionary, project every block onto it, keep the
strongest atoms per block, quantise their coefficients onto the volume column, store the atoms (and their
negations) as PCM, lay the survivors onto channels so they keep their notes, and assemble the module. The
byte size is ``trackmod``'s own model of the file, equal to the bytes it writes; the metrics are measured
against the reconstruction the module reproduces. This is the seam meant for experimentation — set a
config, read the summary, adjust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from trackmod.core.samples.depth import BitDepth
from trackmod.core.songs.playback import Playback
from trackmod.core.timing.timing import Timing
from trackmod.it.timing import row_frames as it_row_frames
from trackmod.limits.violation import Violation
from trackmod.module.protocol import TrackerModule
from trackmod.module.size import SizeReport
from trackmod.spec.levels import MAX_VOLUME
from trackmod.xm.timing import row_frames as xm_row_frames

from audiotokenizer.audio.framing import frame, tukey_window, unframe
from audiotokenizer.audio.io import SAMPLE_RATE, normalise, save_audio
from audiotokenizer.audio.metrics import Metrics, evaluate
from audiotokenizer.coding.assignment import Assignment, assign
from audiotokenizer.coding.cost import bitrate_kbps, kilobytes
from audiotokenizer.coding.quantisation import Quantised, StoredAtoms, quantise, store_atoms
from audiotokenizer.coding.selection import select
from audiotokenizer.dictionary.learned import LearnedDictionary
from audiotokenizer.module.binding import Binding
from audiotokenizer.module.catalog import binding_for
from audiotokenizer.module.samples import POLARITIES
from audiotokenizer.module.song import build_song
from audiotokenizer.pipeline.config import TokenizerConfig
from audiotokenizer.render.openmpt import DEFAULT_INTERPOLATION, Interpolation, render_module


@dataclass(frozen=True)
class CompiledModule:
    """A compiled module with its exact size, measured fidelity, and the reference it approximates."""

    config: TokenizerConfig
    timing: Timing
    module: TrackerModule
    size: SizeReport
    metrics: Metrics
    reference: NDArray[np.float64]
    estimate: NDArray[np.float64]
    n_atoms_used: int
    n_channels_used: int

    @property
    def within_budget(self) -> bool:
        return self.size.total <= self.config.budget_bytes

    @property
    def violations(self) -> tuple[Violation, ...]:
        """Every bound the module breaks, empty when it can be written."""
        return self.module.violations()

    @property
    def writable(self) -> bool:
        """Whether the module writes without raising, which fitting the byte budget does not imply."""
        return not self.violations

    def to_bytes(self) -> bytes:
        return self.module.to_bytes()

    def save(self, path: Path | str) -> Path:
        """Write the module file (``.it`` or ``.xm``) and return its path."""
        destination = Path(path)
        destination.write_bytes(self.to_bytes())
        return destination

    def save_reference(self, path: Path | str) -> Path:
        """Write the reconstruction the module reproduces as a WAV, for A/B against the render."""
        destination = Path(path)
        save_audio(destination, self.estimate, SAMPLE_RATE)
        return destination

    def render(self, *, interpolation: Interpolation = DEFAULT_INTERPOLATION) -> tuple[NDArray[np.float64], int]:
        """Render the module through ``openmpt123`` — the ground-truth playback."""
        return render_module(self.module, sample_rate=SAMPLE_RATE, interpolation=interpolation)

    def summary(self) -> str:
        duration = self.reference.size / SAMPLE_RATE
        fit = "fits" if self.within_budget else "OVER"
        lines = [
            f"{self.config.name}  ({self.config.format}, {self.config.compliance}, {duration:.1f} s)",
            f"  timing        speed {self.timing.speed}, tempo {self.timing.tempo}"
            f"  ->  {self.timing.row_frames} frames/row",
            f"  dictionary    {self.n_atoms_used}/{self.config.n_atoms} atoms used,"
            f" {self.n_channels_used} channels peak",
            f"  size          {kilobytes(self.size.total):.0f} KB  ({fit}"
            f" {kilobytes(self.config.budget_bytes):.0f} KB budget)"
            f"  =  pattern {kilobytes(self.size.patterns):.0f}  pcm {kilobytes(self.size.pcm):.0f}"
            f"  headers {kilobytes(self.size.headers):.0f}",
            f"  bitrate       {bitrate_kbps(self.size.total, duration):.0f} kbps",
            f"  mel LSD       {self.metrics.mel_distance_db:.2f} dB",
            f"  log-spectral  {self.metrics.log_spectral_distance_db:.2f} dB",
            f"  waveform SNR  {self.metrics.waveform_snr_db:.2f} dB",
            f"  envelope err  {self.metrics.envelope_error_db:.2f} dB",
            f"  click ratio   {self.metrics.click_db:.1f} dB  (taper alpha {self.config.taper_alpha:.3f})",
        ]
        return "\n".join(lines)


def _prune_unused(
    quant: Quantised, unit_atoms: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Drop atoms no block plays, returning the reindexed codes, signs, gains and unit atoms."""
    used = np.flatnonzero(np.any(quant.codes > 0, axis=0))
    return quant.codes[:, used], quant.signs[:, used], quant.gains[used], unit_atoms[used]


def _reconstruct(
    sample_no: NDArray[np.int64],
    volume: NDArray[np.int64],
    stored: StoredAtoms,
    *,
    n_samples: int,
) -> NDArray[np.float64]:
    """The signal the module reproduces, read straight off the channel grids the writer emits.

    Each present cell plays sample ``sample_no`` — atom ``sample_no // 2`` at ``+`` for an even slot and
    ``−`` for an odd one — from the volume column, scaled by that atom's stored gain. Summing the channels
    per row and tiling the blocks is exactly what the tracker mixes, so the estimate matches what a
    polarity-sticky assignment actually plays rather than the pre-assignment coefficient signs.
    """
    n_blocks = int(sample_no.shape[0])
    n_atoms = int(stored.pcm.shape[0])
    atom_index = sample_no // POLARITIES
    polarity = np.where(sample_no % POLARITIES == 0, 1.0, -1.0)
    rows, channels = np.nonzero(volume > 0)
    contribution = polarity[rows, channels] * volume[rows, channels] * stored.global_volume[atom_index[rows, channels]]
    weights = np.zeros((n_blocks, n_atoms), dtype=np.float64)
    np.add.at(weights, (rows, atom_index[rows, channels]), contribution)
    blocks = stored.scale * (weights @ stored.pcm) / (MAX_VOLUME * MAX_VOLUME)
    return unframe(blocks, n_samples)


@dataclass(frozen=True)
class Prepared:
    """A timing's fixed groundwork, so a whole sweep at that timing reuses one SVD.

    ``unit_atoms`` holds the dictionary in singular order and ``coef`` the block projections onto it.
    Because the atoms are singular-ordered (Eckart-Young), truncating to the top ``n`` atoms is exactly
    ``unit_atoms[:n]`` / ``coef[:, :n]`` — so a candidate at fewer atoms costs a slice, never a re-SVD.
    Prepare once at the largest pool a search will consider and every smaller candidate is free.
    """

    reference: NDArray[np.float64]
    timing: Timing
    matrix: NDArray[np.float64]
    unit_atoms: NDArray[np.float64]
    coef: NDArray[np.float64]

    @property
    def n_atoms(self) -> int:
        """How many atoms the SVD actually yielded (``≤`` the requested pool for a short signal)."""
        return int(self.unit_atoms.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.matrix.shape[0])


def _prepare(reference: NDArray[np.float64], timing: Timing, n_atoms: int, *, taper_alpha: float) -> Prepared:
    """Frame ``reference`` at ``timing`` and learn the top-``n_atoms`` dictionary and its projections.

    ``taper_alpha`` cosine-tapers each frame's edges before the SVD (see :func:`~audiotokenizer.audio.
    framing.tukey_window`). Because every tapered frame is exactly zero at its endpoints, the row space —
    and so every learned atom — is too, which is what makes a retriggered one-shot sample click-free. One
    elementwise multiply keeps this the same single SVD per timing the sweep relies on.
    """
    matrix = frame(reference, timing.row_frames) * tukey_window(timing.row_frames, taper_alpha)
    unit_atoms = LearnedDictionary().learn(matrix, n_atoms)
    coef = matrix @ unit_atoms.T
    return Prepared(reference=reference, timing=timing, matrix=matrix, unit_atoms=unit_atoms, coef=coef)


def _routable(config: TokenizerConfig, binding: Binding, n_atoms_used: int) -> int:
    """The stored-sample count a pool needs, checked against what the format's keys can reach.

    Raises:
        ValueError: when the pool needs more stored samples than the format routes.
    """
    n_slots = POLARITIES * n_atoms_used
    if n_atoms_used > config.max_atoms:
        raise ValueError(
            f"{n_atoms_used} atoms need {n_slots} stored samples, over the {binding.format} routing limit of "
            f"{POLARITIES * config.max_atoms}; lower n_atoms to at most {config.max_atoms}"
        )
    return n_slots


def _assemble(prepared: Prepared, config: TokenizerConfig) -> CompiledModule:
    """Compile ``prepared`` under ``config``, slicing the shared dictionary to ``config.n_atoms``.

    Everything up to the channel grids is format-agnostic — selection, quantisation, assignment and the
    reconstruction all read the same ``(sample_no, volume)`` slots — so only the binding differs, and
    ``config.format`` picks it. This is the half the CLI and the planner share, with no SVD of its own.

    Raises:
        ValueError: when the pool needs more stored samples than the format can route.
    """
    reference = prepared.reference
    timing = prepared.timing
    binding = binding_for(config.format)
    unit_atoms = prepared.unit_atoms[: config.n_atoms]
    coef = prepared.coef[:, : config.n_atoms]

    selection = select(
        coef, max_per_row=config.max_per_row, min_energy=config.min_energy, persistence=config.persistence
    )
    quant = quantise(np.where(selection.mask, coef, 0.0))
    codes, signs, gains, unit_used = _prune_unused(quant, unit_atoms)
    n_atoms_used = int(unit_used.shape[0])
    _routable(config, binding, n_atoms_used)

    stored = store_atoms(unit_used, gains, bits=config.pcm_bits)
    n_channels = min(config.max_per_row, max(n_atoms_used, 1))
    assignment = assign(codes, signs, n_channels=n_channels, sticky_threshold=config.polarity_sticky)

    song = build_song(
        binding,
        stored,
        assignment,
        name=config.name,
        playback=Playback(speed=timing.speed, tempo=timing.tempo),
        rate=SAMPLE_RATE,
        depth=BitDepth(config.pcm_bits),
    )
    module = binding.module(song, compliance=config.compliance)
    estimate = _reconstruct(assignment.sample_no, assignment.volume, stored, n_samples=reference.size)
    return CompiledModule(
        config=config,
        timing=timing,
        module=module,
        size=module.size(),
        metrics=evaluate(reference, estimate, block_len=timing.row_frames),
        reference=reference,
        estimate=estimate,
        n_atoms_used=n_atoms_used,
        n_channels_used=_peak_channels(assignment),
    )


def _peak_channels(assignment: Assignment) -> int:
    """The most channels any single row plays at once."""
    if not assignment.volume.size:
        return 0
    return int(np.max(np.count_nonzero(assignment.volume > 0, axis=1)))


def timing_for(config: TokenizerConfig, *, frame_rate: int) -> Timing:
    """The row length ``config``'s speed and tempo give, read on the chosen format's clock.

    Both formats share the tick clock and differ only in how wide a tempo their header stores, which is
    exactly what decides how short a row a caller can ask for.

    Raises:
        ValueError: when the speed and tempo give a row that is not a whole number of frames.
    """
    frames = xm_row_frames if config.format is config.format.XM else it_row_frames
    return Timing(
        speed=config.speed,
        tempo=config.tempo,
        row_frames=frames(config.speed, config.tempo, frame_rate=frame_rate),
    )


def compile_signal(signal: NDArray[np.float64], config: TokenizerConfig) -> CompiledModule:
    """Compile ``signal`` (mono float at 44100 Hz) into a module of ``config.format``.

    Raises:
        ValueError: when the pool needs more stored samples than the format can route, or when the timing
            gives a row that is not a whole number of frames.
    """
    reference = normalise(np.asarray(signal, dtype=np.float64).ravel())
    timing = timing_for(config, frame_rate=SAMPLE_RATE)
    return _assemble(_prepare(reference, timing, config.n_atoms, taper_alpha=config.taper_alpha), config)
