# Architecture & Ownership

AudioTokenizer approximates a signal with a learned dictionary of short samples and compiles it into a
tracker module — Impulse Tracker (`.it`) or FastTracker 2 (`.xm`) — under a hard byte budget, **2 MB** by
default. The whole tool solves one equation, `M ≈ C·D`: `M` is the block matrix (one row per block of the
signal), `D` is a dictionary of unit-norm atoms stored as tracker samples, and `C` is the per-block
coefficient matrix quantised onto the 6-bit volume column.

Everything about the *file formats* lives in
[`trackmod`](https://github.com/JakimPL/TrackMod/blob/main/docs/overview.md), a standalone library that
holds one format-agnostic song model and binds it to both formats. It is taken here as a git submodule at
`trackmod/`, so its documents are also readable at `trackmod/docs/` in a checkout. This document says which
part of the codec owns what, so shared logic has one home and new code lands in the right place.

## Package map (`audiotokenizer/`)

| Module / subpackage | Owns | Depends on |
|---|---|---|
| `audio/` | The signal boundary: `io` (WAV load/save at 44100 Hz), `framing` (signal ↔ `(n_blocks, block_len)` matrix, plus the Tukey taper that zeros atom edges), `metrics` (SNR, log-spectral, mel, envelope and click distance — the perceptual scorers). | numpy, scipy, soundfile |
| `dictionary/` | The atom vocabulary: the `Dictionary` protocol (`learn(blocks, n_atoms) -> unit-norm atoms`) and `learned` (its SVD implementation — Eckart–Young optimal for the block set). Another vocabulary lands here behind the same protocol. | numpy |
| `coding/` | The merge, all of it format-agnostic: `quantisation` (coefficients → volume column, atoms → PCM plus per-atom gain), `selection` (per-row atom choice), `assignment` (chosen atoms → channels, so a surviving atom keeps its note), `cost` (budget arithmetic and the planner's lower bound). | `trackmod.spec`, `module.it`, numpy |
| `module/` | The seam between the codec and `trackmod`: `format` (which formats exist), `binding` (the `Binding` protocol — the four choices the formats disagree on), `it`/`xm` (one binding each), `catalog` (name → binding), `routing`, `slicing`, `samples`, `instruments`, `patterns`, `song` (the assembly, written once against the protocol). | `trackmod`, `coding`, numpy |
| `pipeline/` | The end-to-end conversion: `config` (the frozen Pydantic knobs, validated against the format's own limits), `compiler` (signal → cells → module → bytes, returning a result that saves/renders/summarises), `planner` (search the lattice for the best quality under the budget). | everything above |
| `render/` | External playback tooling behind a probed boundary: `openmpt` (render a module or its bytes through `openmpt123`) and `xmodits` (rip a module's samples back out). | `trackmod.module.protocol`, `audio.io` |
| `cli.py` | Entry point: parse arguments, resolve the format, build the config, run the compiler or the planner, print what it spent. | `pipeline`, `module` |

## Rules

1. **Format knowledge lives in `trackmod`.** Record layouts, packers, parsers, size models and the limit
   tables are the library's, and it is a peer of this package rather than a part of it. A format constant
   is imported from its `spec/` module; nothing here re-derives one.
2. **`module/` is the only place that knows there is more than one format.** Everything above it reasons
   in atoms, coefficients and channel slots; everything below it in notes, instruments and patterns. The
   four choices that genuinely differ are named in the `Binding` protocol — how a dictionary is routed,
   where an atom's gain lives, whether a cell restates its instrument, and how a song is cut into patterns
   — so the assembly itself is written once.
3. **Layering runs strictly downward.** `audio` → `dictionary` → `coding` → `module` → `pipeline` → `cli`,
   with `render` alongside for tooling. Nothing below `pipeline` knows what a WAV file is; nothing below
   `pipeline` reads the config object. Config is validated at the boundary and threads down as explicit
   arguments.
4. **One responsibility per module.** Keep data shapes, algorithms and serialisation separated. When a
   group of related types would otherwise become a bag of classes, make it a small subpackage.
5. **No pass-through re-exports.** An `__init__` exposes only its own subtree's public API; import shared
   helpers directly from their owner.
6. **Compliance is a value, and it comes from the format.** `Compliance.CANONICAL` holds a module to what
   the tracker it names actually honoured; `Compliance.EXTENDED` holds it to what the record layout can
   physically store. Both are read off the format's limit table rather than repeated here, which is what
   keeps a caller from asking Impulse Tracker for a 16-bit tempo its header has one byte for. See
   [`trackmod`'s `limits.md`](https://github.com/JakimPL/TrackMod/blob/main/docs/limits.md).
7. **External tools sit behind a typed, probed boundary.** `render/openmpt.py` exposes
   `openmpt123_available()`, and `render/xmodits.py` is a dev-only dependency. Callers degrade gracefully
   instead of branching on the environment.

## The size model, and the one bound that is not it

A module's exact size is `trackmod`'s own model of it — `TrackerModule.size()`, pinned to the writers by
the keystone test `module.size().total == len(module.to_bytes())` — so the codec re-derives nothing.
`coding/cost.py` keeps only the arithmetic the budget is quoted in (kilobytes, kilobits per second) and
one thing that model cannot supply: `cost_lower_bound`, a floor on a candidate's size from cell counts
alone.

The planner needs that floor because assembling a candidate costs a per-row assignment and a full metric
pass, so a hopeless config has to be rejected before it is built. A packed Impulse Tracker pattern spends
at least a terminator per row and a marker plus a volume byte per present cell — note, instrument and mask
bytes only ever add to that — while the PCM and the record overhead follow exactly from the sample and
pattern counts. The bound therefore stays at or below the true size, so a candidate it rejects could not
have fit and the sweep loses nothing by trusting it.

## Where the two formats part ways

Both write the same song; the codec pays for the difference in four places, all of them behind `Binding`.

| | Impulse Tracker | FastTracker 2 |
|---|---|---|
| **Routing** | one instrument covers 120 keys, from the lowest | one covers 16, starting at C-2 |
| **An atom's gain** | rides in the sample's global-volume lattice, which the volume column multiplies | baked into the waveform, since the volume column *overrides* the sample volume byte |
| **Restating the instrument** | on every played cell, so each note re-applies its own atom's gain | only where the channel is not already carrying it |
| **Pattern split** | as few patterns as the 200-row ceiling allows, because per-channel reuse memory resets at each boundary | as many as the 256-entry order table holds, because the packing keeps no memory and shorter patterns stay further from the 16-bit length field |

## Tests

`tests/` mirrors this map, plus `tests/render/`, which answers to the outside world by driving the real
tracker. `test_openmpt.py` correlates a render against the reconstruction the codec predicts, and
`test_limits.py` plays each extended bound — 127 IT channels, 192 XM channels, XM at 441 BPM — asserting
the rendered audio lasts as long as the row clock says, which a quietly clamped field would fail.
