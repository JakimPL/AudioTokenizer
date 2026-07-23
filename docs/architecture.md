# Architecture & Ownership

AudioTokenizer approximates a signal with a learned dictionary of short samples and compiles it into an
Impulse Tracker (`.IT`) module under a hard byte budget — **2 MB** by default. The whole tool solves one
equation, `M ≈ C·D`: `M` is the block matrix (one row per block of the signal), `D` is a dictionary of
unit-norm atoms stored as tracker samples, and `C` is the per-block coefficient matrix quantised onto the
6-bit volume column. This document says which part of the package owns what, so shared logic has one home
and new code lands in the right place.

## Package map (`audiotokenizer/`)

| Module / subpackage | Owns | Depends on |
|---|---|---|
| `audio/` | The signal boundary: `io` (WAV load/save at 44100 Hz), `framing` (signal ↔ `(n_blocks, block_len)` matrix), `metrics` (SNR, log-spectral, mel and envelope distance — the perceptual scorers). | numpy, scipy, soundfile |
| `dictionary/` | The atom vocabulary: the `Dictionary` protocol (`learn(blocks, n_atoms) -> unit-norm atoms`) and `learned` (its SVD implementation — Eckart–Young optimal for the block set). The sine bank lands here behind the same protocol. | numpy |
| `coding/` | The merge: `cost` (the byte-exact `.IT` size model), `quantisation` (coefficients → volume column, atoms → PCM), `selection` (per-row atom choice), `assignment` (chosen atoms → channels, minimising pattern bytes). | `it.spec`, numpy |
| `it/` | The Impulse Tracker format boundary: `spec` (every format constant — magic, masks, flags, limits, record sizes), `format` (declarative record layout), `samples`/`instruments`/`patterns`/`module` (the binary serializer), `timing` (tempo/speed ↔ the integer frame lattice), `pitch` (unity-playback C5Speed), `render` (openmpt123 wrapper), `reader` (xmodits round-trip). | `coding.cost` (size constants live in `spec`), numpy, soundfile |
| `pipeline/` | The end-to-end conversion: `config` (the frozen Pydantic knobs), `compiler` (signal → cells → `ITModule` → bytes, returning a project that saves/renders/summarises), `planner` (search the lattice for the best quality under the budget). | everything below |
| `cli.py` | Entry point: parse arguments, build the config, run the compiler, print what it spent. | `pipeline` |

## Rules

1. **Shared primitives have one home.** Every IT format constant lives in `it/spec.py`; the record layout in
   `it/format.py`; the byte-size model in `coding/cost.py`. Import from the owner — do not re-derive. The
   cost model and the pattern writer must agree byte-for-byte; the `pack_pattern == cost.pattern_bytes` test
   is the keystone that pins them together.
2. **Layering runs strictly downward.** `audio` → `dictionary` → `coding` → `it` → `pipeline` → `cli`.
   Nothing below `pipeline` knows what a WAV file is; nothing below `pipeline` reads the config object.
   Config loads at the boundary and threads down as explicit arguments.
3. **One responsibility per module.** Keep data shapes, algorithms, and serialization separated. When a
   group of related types would otherwise become a bag of classes, make it a small subpackage.
4. **No pass-through re-exports.** An `__init__` exposes only its own subtree's public API; do not add
   re-export shims so other modules can import through them. Import shared helpers directly from their owner.
5. **Two IT profiles, one writer.** `strict` caps channels at 64 (canonical IT, which masks the pattern
   channel byte to 6 bits); `hacked` allows up to 127 (OpenMPT masks to 7 bits, verified by rendering a
   probe module). The profile is a value threaded from config; the writer widens its channel guard with it.
6. **External tools sit behind a typed, probed boundary.** `openmpt123` is wrapped in `it/render.py` with a
   runtime `openmpt123_available()` probe; `xmodits` in `it/reader.py`. Callers degrade gracefully instead
   of branching on the environment.

## The byte model, stated once

A packed pattern row is a list of present channels terminated by a zero byte; an absent cell costs nothing.
A present cell costs a channel marker plus a volume byte (2 bytes), plus a note byte only when the atom
differs from that channel's previous present row, plus a mask byte only when the cell's mask differs from
the channel's last. Impulse Tracker's `0x10` mask bit re-triggers a channel's previous note for free, so a
channel that keeps its atom settles to two bytes a row. Atoms are unlooped and exactly one row long, so a
silent channel is free and a polarity flip (which re-points a cell at the `−s` sample) pays a note byte.
This is the whole basis of the budget, and it lives in `coding/cost.py`.
