"""Compile a signal into an Impulse Tracker module end to end, and report exactly what it cost.

The pipeline runs one pass: frame the signal, learn a dictionary, project every block onto it, keep the
strongest atoms per block, quantise their coefficients onto the volume column, store the atoms (and their
negations) as PCM, lay the survivors onto channels so they keep their notes, and assemble the module. The
byte size is the byte-exact model, equal to the written file; the metrics are measured against the
reconstruction the module reproduces. This is the seam meant for experimentation — set a config, read the
summary, adjust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.audio.framing import frame, tukey_window, unframe
from audiotokenizer.audio.io import SAMPLE_RATE, normalise, save_audio
from audiotokenizer.audio.metrics import Metrics, evaluate
from audiotokenizer.coding.assignment import Assignment, assign
from audiotokenizer.coding.cost import Cost, module_bytes, pattern_slices
from audiotokenizer.coding.quantisation import Quantised, StoredAtoms, quantise, store_atoms
from audiotokenizer.coding.selection import select
from audiotokenizer.coding.xm_cost import xm_module_bytes, xm_writable
from audiotokenizer.dictionary.learned import LearnedDictionary
from audiotokenizer.it.instruments import ITInstrument, fixed_c5_note_map
from audiotokenizer.it.module import ITModule, write_it_module
from audiotokenizer.it.patterns import ITPattern, ITPlayback
from audiotokenizer.it.render import Interpolation
from audiotokenizer.it.render import render_module as render_it_module
from audiotokenizer.it.samples import ITSample
from audiotokenizer.it.spec import KEYBOARD_NOTES, MAX_SAMPLES, MAX_VOLUME
from audiotokenizer.it.timing import Timing, row_frames
from audiotokenizer.pipeline.config import TokenizerConfig
from audiotokenizer.xm.instruments import XMInstrument, play_note_for
from audiotokenizer.xm.module import XMModule, write_xm_module
from audiotokenizer.xm.patterns import XMPattern, XMPlayback
from audiotokenizer.xm.render import render_module as render_xm_module
from audiotokenizer.xm.samples import XMSample, relative_note_for
from audiotokenizer.xm.spec import MAX_PATTERN_BYTES as XM_MAX_PATTERN_BYTES
from audiotokenizer.xm.spec import MAX_PATTERNS as XM_MAX_PATTERNS
from audiotokenizer.xm.spec import MAX_ROWS as XM_MAX_ROWS
from audiotokenizer.xm.spec import MAX_SAMPLES_TOTAL as XM_MAX_SAMPLES
from audiotokenizer.xm.spec import SAMPLES_PER_INSTRUMENT

_GLOBAL_VOLUME: Final = 128
_MIX_VOLUME: Final = 48
_POLARITIES: Final = 2  # each atom stores its positive and negative sample


@dataclass(frozen=True)
class CompiledModule:
    """A compiled module with its exact size, measured fidelity, and the reference it approximates."""

    config: TokenizerConfig
    timing: Timing
    module: ITModule | XMModule
    cost: Cost
    metrics: Metrics
    reference: NDArray[np.float64]
    estimate: NDArray[np.float64]
    n_atoms_used: int
    n_channels_used: int

    @property
    def within_budget(self) -> bool:
        return self.cost.total <= self.config.budget_bytes

    def to_bytes(self) -> bytes:
        if isinstance(self.module, XMModule):
            return write_xm_module(self.module)
        return write_it_module(self.module)

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

    def render(self, *, interpolation: Interpolation = "sinc") -> tuple[NDArray[np.float64], int]:
        """Render the module through ``openmpt123`` — the ground-truth playback."""
        if isinstance(self.module, XMModule):
            return render_xm_module(self.module, sample_rate=SAMPLE_RATE, interpolation=interpolation)
        return render_it_module(self.module, sample_rate=SAMPLE_RATE, interpolation=interpolation)

    def summary(self) -> str:
        duration = self.reference.size / SAMPLE_RATE
        fit = "fits" if self.within_budget else "OVER"
        lines = [
            f"{self.config.name}  ({self.config.format}, {self.config.profile}, {duration:.1f} s)",
            f"  timing        speed {self.timing.speed}, tempo {self.timing.tempo}"
            f"  ->  {self.timing.row_frames} frames/row",
            f"  dictionary    {self.n_atoms_used}/{self.config.n_atoms} atoms used,"
            f" {self.n_channels_used} channels peak",
            f"  size          {self.cost.kilobytes:.0f} KB  ({fit} {self.config.budget_bytes / 1024:.0f} KB"
            f" budget)  =  pattern {self.cost.pattern / 1024:.0f}  pcm {self.cost.pcm / 1024:.0f}"
            f"  headers {self.cost.headers / 1024:.0f}",
            f"  bitrate       {self.cost.bitrate_kbps(duration):.0f} kbps",
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


def _build_samples(stored: StoredAtoms, *, bits: int) -> tuple[ITSample, ...]:
    """Two stored samples per atom — its positive PCM and its negation — in slot order.

    Both carry the atom's gain in their global-volume field, so the peak-normalized PCM plays back at the
    atom's amplitude.
    """
    samples: list[ITSample] = []
    for index, (atom, gain_volume) in enumerate(zip(stored.pcm, stored.global_volume)):
        volume = int(gain_volume)
        samples.append(
            ITSample(name=f"atom{index}+", pcm=atom, depth_bits=bits, c5speed=SAMPLE_RATE, global_volume=volume)
        )
        samples.append(
            ITSample(name=f"atom{index}-", pcm=-atom, depth_bits=bits, c5speed=SAMPLE_RATE, global_volume=volume)
        )
    return tuple(samples)


def _build_instruments(n_slots: int) -> tuple[ITInstrument, ...]:
    """One instrument per 120 sample slots, each routing its keys 1-based to those samples at unity pitch."""
    instruments: list[ITInstrument] = []
    for start in range(0, n_slots, KEYBOARD_NOTES):
        stop = min(start + KEYBOARD_NOTES, n_slots)
        assignments = {slot - start: slot + 1 for slot in range(start, stop)}
        instruments.append(ITInstrument(name=f"dict{start // KEYBOARD_NOTES}", note_map=fixed_c5_note_map(assignments)))
    return tuple(instruments)


def _build_patterns(assignment: Assignment) -> tuple[ITPattern, ...]:
    """Slice the channel grids into ITTECH's 200-row patterns."""
    n_rows = assignment.sample_no.shape[0]
    return tuple(
        ITPattern(
            rows=stop - start,
            sample_no=assignment.sample_no[start:stop],
            volume=assignment.volume[start:stop],
        )
        for start, stop in pattern_slices(n_rows)
    )


def _build_it(
    config: TokenizerConfig, stored: StoredAtoms, assignment: Assignment, *, n_slots: int, block_len: int
) -> tuple[ITModule, Cost]:
    """Assemble the IT module and its byte-exact cost from the shared stored atoms and channel grids."""
    n_rows = int(assignment.sample_no.shape[0])
    module = ITModule(
        name=config.name,
        samples=_build_samples(stored, bits=config.pcm_bits),
        instruments=_build_instruments(n_slots),
        patterns=_build_patterns(assignment),
        orders=tuple(range(len(pattern_slices(n_rows)))),
        playback=ITPlayback(
            speed=config.speed, tempo=config.tempo, global_volume=_GLOBAL_VOLUME, mix_volume=_MIX_VOLUME
        ),
    )
    cost = module_bytes(
        assignment.sample_no,
        assignment.volume,
        n_stored_samples=n_slots,
        pcm_frames=block_len,
        bits_per_frame=config.pcm_bits,
    )
    return module, cost


def _xm_rows_per_pattern(n_rows: int) -> int:
    """The smallest uniform pattern height that keeps the song inside XM's 256-pattern order table.

    Smaller patterns pack under the u16 length more readily, so slicing at exactly the height the order
    table forces is the most feasible choice. A song that would still need patterns taller than
    :data:`XM_MAX_ROWS` cannot be written as XM at all — the feasibility wall IT does not have.
    """
    rows_per_pattern = max(1, -(-n_rows // XM_MAX_PATTERNS))
    if rows_per_pattern > XM_MAX_ROWS:
        raise ValueError(
            f"{n_rows} rows need {rows_per_pattern}-row patterns, over XM's {XM_MAX_ROWS}-row limit across "
            f"{XM_MAX_PATTERNS} patterns; use a longer row (lower tempo) or a shorter signal to export XM"
        )
    return rows_per_pattern


def _build_xm_samples(stored: StoredAtoms, *, bits: int) -> tuple[XMSample, ...]:
    """Two stored samples per atom — its positive PCM and its negation — each tuned to play native on its key.

    The atom's gain rides in the sample volume byte (XM's counterpart of IT's sample global-volume), and a
    per-sample relative-note re-tunes the key the pattern triggers back to 44100 Hz.
    """
    samples: list[XMSample] = []
    for index, (atom, gain_volume) in enumerate(zip(stored.pcm, stored.global_volume)):
        volume = int(gain_volume)
        for sign, suffix in ((1.0, "+"), (-1.0, "-")):
            slot = _POLARITIES * index + (0 if sign > 0 else 1)
            play_note = play_note_for(slot % SAMPLES_PER_INSTRUMENT)
            samples.append(
                XMSample(
                    name=f"atom{index}{suffix}",
                    pcm=sign * atom,
                    depth_bits=bits,
                    volume=volume,
                    relative_note=relative_note_for(play_note),
                )
            )
    return tuple(samples)


def _build_xm_instruments(n_slots: int) -> tuple[XMInstrument, ...]:
    """One instrument per 16 sample slots (8 atoms x 2 polarities), each routing its keys to those samples."""
    instruments: list[XMInstrument] = []
    for start in range(0, n_slots, SAMPLES_PER_INSTRUMENT):
        count = min(SAMPLES_PER_INSTRUMENT, n_slots - start)
        instruments.append(XMInstrument(name=f"dict{start // SAMPLES_PER_INSTRUMENT}", n_samples=count))
    return tuple(instruments)


def _build_xm_patterns(assignment: Assignment, rows_per_pattern: int) -> tuple[XMPattern, ...]:
    """Slice the channel grids into patterns of ``rows_per_pattern`` rows (chosen for feasibility)."""
    n_rows = assignment.sample_no.shape[0]
    return tuple(
        XMPattern(
            rows=stop - start,
            sample_no=assignment.sample_no[start:stop],
            volume=assignment.volume[start:stop],
        )
        for start, stop in pattern_slices(n_rows, rows_per_pattern=rows_per_pattern)
    )


def _build_xm(
    config: TokenizerConfig,
    stored: StoredAtoms,
    assignment: Assignment,
    *,
    n_slots: int,
    block_len: int,
    n_channels: int,
) -> tuple[XMModule, Cost]:
    """Assemble the XM module and its byte-exact cost, enforcing the feasibility wall with a clear error."""
    n_rows = int(assignment.sample_no.shape[0])
    rows_per_pattern = _xm_rows_per_pattern(n_rows)
    cost = xm_module_bytes(
        assignment.sample_no,
        assignment.volume,
        n_stored_samples=n_slots,
        pcm_frames=block_len,
        bits_per_frame=config.pcm_bits,
        rows_per_pattern=rows_per_pattern,
    )
    if not xm_writable(cost, rows_per_pattern=rows_per_pattern):
        raise ValueError(
            f"the densest XM pattern packs to {cost.max_pattern} bytes, over the {XM_MAX_PATTERN_BYTES}-byte "
            f"u16 limit even at {rows_per_pattern} rows/pattern; use fewer channels or a longer row for XM"
        )
    module = XMModule(
        name=config.name,
        samples=_build_xm_samples(stored, bits=config.pcm_bits),
        instruments=_build_xm_instruments(n_slots),
        patterns=_build_xm_patterns(assignment, rows_per_pattern),
        orders=tuple(range(len(pattern_slices(n_rows, rows_per_pattern=rows_per_pattern)))),
        playback=XMPlayback(speed=config.speed, tempo=config.tempo),
        channels=n_channels,
    )
    return module, cost


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
    atom_index = sample_no // 2
    polarity = np.where(sample_no % 2 == 0, 1.0, -1.0)
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


def _assemble(prepared: Prepared, config: TokenizerConfig) -> CompiledModule:
    """Compile ``prepared`` under ``config``, slicing the shared dictionary to ``config.n_atoms``.

    Everything up to the channel grids is format-agnostic — selection, quantisation, assignment and the
    reconstruction all read the same ``(sample_no, volume)`` slots — so only the module and its byte model
    differ: ``config.format`` picks the IT or XM builder. This is the half the CLI and the planner share,
    with no SVD of its own.

    Raises:
        ValueError: when the pool needs more stored samples than the format can route, or (XM only) when the
            song trips the feasibility wall — too many patterns, or a pattern over the u16 length.
    """
    reference = prepared.reference
    timing = prepared.timing
    block_len = timing.row_frames
    unit_atoms = prepared.unit_atoms[: config.n_atoms]
    coef = prepared.coef[:, : config.n_atoms]

    selection = select(
        coef, max_per_row=config.max_per_row, min_energy=config.min_energy, persistence=config.persistence
    )
    quant = quantise(np.where(selection.mask, coef, 0.0))
    codes, signs, gains, unit_used = _prune_unused(quant, unit_atoms)
    n_atoms_used = int(unit_used.shape[0])
    n_slots = _POLARITIES * n_atoms_used
    routing_limit = XM_MAX_SAMPLES if config.format == "xm" else MAX_SAMPLES
    if n_slots > routing_limit:
        raise ValueError(
            f"{n_atoms_used} atoms need {n_slots} stored samples, over the {config.format} routing limit of "
            f"{routing_limit}; lower n_atoms to at most {routing_limit // _POLARITIES}"
        )

    stored = store_atoms(unit_used, gains, bits=config.pcm_bits)
    n_channels = min(config.max_per_row, max(n_atoms_used, 1))
    assignment = assign(codes, signs, n_channels=n_channels, sticky_threshold=config.polarity_sticky)

    module: ITModule | XMModule
    if config.format == "xm":
        module, cost = _build_xm(
            config, stored, assignment, n_slots=n_slots, block_len=block_len, n_channels=n_channels
        )
    else:
        module, cost = _build_it(config, stored, assignment, n_slots=n_slots, block_len=block_len)
    estimate = _reconstruct(assignment.sample_no, assignment.volume, stored, n_samples=reference.size)
    n_channels_used = int(np.max(np.count_nonzero(assignment.volume > 0, axis=1))) if assignment.volume.size else 0
    return CompiledModule(
        config=config,
        timing=timing,
        module=module,
        cost=cost,
        metrics=evaluate(reference, estimate, block_len=block_len),
        reference=reference,
        estimate=estimate,
        n_atoms_used=n_atoms_used,
        n_channels_used=n_channels_used,
    )


def compile_signal(signal: NDArray[np.float64], config: TokenizerConfig) -> CompiledModule:
    """Compile ``signal`` (mono float at 44100 Hz) into an IT or XM module under ``config.format``.

    Raises:
        ValueError: when the pool needs more stored samples than the format can route, or (XM only) when the
            song trips the feasibility wall — too many patterns, or a pattern over the u16 length.
    """
    reference = normalise(np.asarray(signal, dtype=np.float64).ravel())
    timing = Timing(
        config.speed,
        config.tempo,
        row_frames(
            config.speed,
            config.tempo,
            frame_rate=SAMPLE_RATE,
            max_tempo=config.max_tempo,
        ),
    )
    return _assemble(_prepare(reference, timing, config.n_atoms, taper_alpha=config.taper_alpha), config)
