# AudioTokenizer

Approximate an audio signal under a byte budget and compile it into an **Impulse Tracker** (`.it`)
module. The signal is framed into rows of one tracker tick; a small dictionary of short samples is
learned from those rows (the leading right-singular vectors of the block matrix); each row is written as a
handful of those samples triggered from the pattern's volume column. The result is a real tracker module
that plays in OpenMPT and reconstructs the input, kept under a target size (**2 MB** by default).

The whole encoder is one downward-layered pass — `audio → dictionary → coding → it → pipeline → cli`.
See `docs/architecture.md` for the package map and `docs/guidelines.md` for the coding style.

## Install

The file formats live in [`trackmod`](https://github.com/JakimPL/TrackMod), taken here as a git submodule
so a checkout pins the exact revision this project was built against.

The project itself is managed with
[`uv`](https://docs.astral.sh/uv/):

```shell
git submodule update --init
uv sync
```

A fresh clone can do both in one step with `git clone --recurse-submodules`.

## Usage

```shell
uv run audiotokenizer audio/song.wav -o song.it
```

`python -m audiotokenizer ...` is an equivalent entry point. Input must be **mono, 44100 Hz**; anything else
is rejected rather than silently resampled. Every run prints what it spent:

```text
song  (hacked, 169.2 s)
  timing        speed 1, tempo 125  ->  882 frames/row
  dictionary    88/88 atoms used, 88 channels peak
  size          1943 KB  (fits 2048 KB budget)  =  pattern 1775  pcm 152  headers 16
  bitrate       94 kbps
  mel LSD       7.27 dB
  log-spectral  ...
  waveform SNR  19.31 dB
  envelope err  0.69 dB
  wrote         song.it
```

### Options

| flag | default | meaning |
|---|---|---|
| `input` | — | input WAV, mono 44100 Hz |
| `-o, --output` | `<input>.it` | output module path |
| `--strict` | off | canonical 64-channel IT (off = 127-channel "hacked" IT, OpenMPT only) |
| `--tempo` | `125` | IT tempo; sets the row length in frames (speed 1, tempo 125 → 882) |
| `--atoms` | `88` | dictionary size (the atom pool / peak channel count) |
| `--min-energy` | `0.0` | drop per-row projections weaker than this |
| `--pcm-bits` | `8` | stored sample depth (`8` or `16`) |
| `--budget` | `2097152` | byte budget to report `fits`/`OVER` against |
| `--wav PATH` | — | also write the reconstruction as a WAV, for A/B listening |

### Profiles

- **hacked** (default) — up to **127 channels**. Uses OpenMPT's 7-bit channel mask; plays correctly in
  libopenmpt/OpenMPT but is beyond canonical Impulse Tracker.
- **strict** (`--strict`) — **64 channels**, a faithful canonical `.it` for any conforming player.

## Measuring fidelity

Every run scores the reconstruction against the input and prints the numbers above; they are computed by
`audiotokenizer.audio.metrics.evaluate(reference, estimate)`:

- **mel LSD** — RMS log error over 40 mel bands (dB, lower better). The headline perceptual metric.
- **log-spectral** — RMS log-magnitude STFT error (dB).
- **waveform SNR** — gain-aligned time-domain signal-to-noise (dB, higher better).
- **envelope err** — RMS log error of the short-time energy envelope (dB); catches attack smearing a
  magnitude spectrogram misses.

All are gain-aligned, so they report *shape* agreement, not level. These score the surrogate
reconstruction (the exact signal the module encodes). For **ground truth**, render the written `.it`
through OpenMPT and compare that: `--wav out.wav` writes the surrogate for A/B, and
`CompiledModule.render()` renders the module through `openmpt123` (if it is on `PATH`). The mel distance
matches the render closely; surrogate SNR / envelope are mildly optimistic because the real player adds
interpolation and per-note volume ramping the block-edge surrogate ignores.

## Development

```shell
make test     # pytest
make lint     # mypy (strict) + pylint
make format   # isort + black
```
