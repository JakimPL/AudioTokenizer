"""Perceptual scorers for a reconstruction: spectral, mel-band, waveform and short-time envelope error.

Envelope error is the metric that exposes attack smearing a magnitude spectrogram misses, so it rides
alongside the spectral distances; all are gain-aligned, reporting shape agreement rather than level.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from audiotokenizer.audio.io import SAMPLE_RATE

__all__ = [
    "stft_magnitude",
    "snr_db",
    "spectral_snr_db",
    "log_spectral_distance",
    "mel_distance_db",
    "envelope_error_db",
    "crest_factor",
    "Metrics",
    "evaluate",
]

_LSD_FLOOR_DB: Final = 80.0
_MEL_FLOOR_DB: Final = 80.0
_MEL_BANDS: Final = 40
_ENVELOPE_WINDOW: Final = 128  # 2.9 ms at 44100 Hz


def _match_lengths(
    reference: NDArray[np.float64], estimate: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = min(reference.size, estimate.size)
    return (
        np.asarray(reference, dtype=np.float64)[:n],
        np.asarray(estimate, dtype=np.float64)[:n],
    )


def _optimal_gain(reference: NDArray[np.float64], estimate: NDArray[np.float64]) -> float:
    denominator = float(estimate @ estimate)
    return float(reference @ estimate) / denominator if denominator > 0.0 else 0.0


def stft_magnitude(signal: NDArray[np.float64], *, n_fft: int = 2048, hop: int = 512) -> NDArray[np.float64]:
    """``(n_bins, n_frames)`` magnitude STFT with a Hann window.

    The analysis window is fixed, independent of the block length being tested,
    so numbers stay comparable across different ``T``.
    """
    flat = np.asarray(signal, dtype=np.float64).ravel()
    if flat.size < n_fft:
        flat = np.pad(flat, (0, n_fft - flat.size))
    window = np.hanning(n_fft + 1)[:n_fft]
    starts = np.arange(0, flat.size - n_fft + 1, hop)
    frames = np.stack([flat[s : s + n_fft] for s in starts], axis=1)
    return np.abs(np.fft.rfft(frames * window[:, None], axis=0))


def snr_db(reference: NDArray[np.float64], estimate: NDArray[np.float64], *, align_gain: bool = True) -> float:
    """Waveform signal-to-noise ratio in dB.

    With ``align_gain`` the estimate is scaled optimally first, so the number
    reports waveform shape agreement rather than level agreement.
    """
    reference, estimate = _match_lengths(reference, estimate)
    if align_gain:
        estimate = estimate * _optimal_gain(reference, estimate)
    noise = float(np.sum((reference - estimate) ** 2))
    signal = float(np.sum(reference**2))
    if noise <= 0.0:
        return float("inf")
    return 10.0 * np.log10(signal / noise) if signal > 0.0 else float("-inf")


def spectral_snr_db(
    reference: NDArray[np.float64],
    estimate: NDArray[np.float64],
    *,
    n_fft: int = 2048,
    hop: int = 512,
) -> float:
    """Signal-to-error ratio between magnitude spectrograms, in dB."""
    reference, estimate = _match_lengths(reference, estimate)
    ref_mag = stft_magnitude(reference, n_fft=n_fft, hop=hop)
    est_mag = stft_magnitude(estimate, n_fft=n_fft, hop=hop)
    denominator = float(np.sum(est_mag * est_mag))
    if denominator > 0.0:
        est_mag = est_mag * (float(np.sum(ref_mag * est_mag)) / denominator)
    noise = float(np.sum((ref_mag - est_mag) ** 2))
    signal = float(np.sum(ref_mag**2))
    if noise <= 0.0:
        return float("inf")
    return 10.0 * np.log10(signal / noise) if signal > 0.0 else float("-inf")


def log_spectral_distance(
    reference: NDArray[np.float64],
    estimate: NDArray[np.float64],
    *,
    n_fft: int = 2048,
    hop: int = 512,
) -> float:
    """RMS difference of log-magnitude spectrograms, in dB (lower is better)."""
    reference, estimate = _match_lengths(reference, estimate)
    ref_mag = stft_magnitude(reference, n_fft=n_fft, hop=hop)
    est_mag = stft_magnitude(estimate, n_fft=n_fft, hop=hop)
    denominator = float(np.sum(est_mag * est_mag))
    if denominator > 0.0:
        est_mag = est_mag * (float(np.sum(ref_mag * est_mag)) / denominator)
    peak = float(np.max(ref_mag)) if ref_mag.size else 0.0
    floor = peak * 10.0 ** (-_LSD_FLOOR_DB / 20.0) if peak > 0.0 else 1e-12
    ref_db = 20.0 * np.log10(np.maximum(ref_mag, floor))
    est_db = 20.0 * np.log10(np.maximum(est_mag, floor))
    return float(np.sqrt(np.mean((ref_db - est_db) ** 2)))


def _mel_filterbank(n_fft: int, sample_rate: int, n_bands: int) -> NDArray[np.float64]:
    def to_mel(hz: float) -> float:
        return 2595.0 * float(np.log10(1.0 + hz / 700.0))

    def to_hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    edges = to_hz(np.linspace(to_mel(40.0), to_mel(sample_rate / 2), n_bands + 2))
    bins = np.floor((n_fft + 1) * edges / sample_rate).astype(int)
    bank = np.zeros((n_bands, n_fft // 2 + 1), dtype=np.float64)
    for band in range(n_bands):
        low, mid, high = bins[band], bins[band + 1], min(bins[band + 2], n_fft // 2)
        if mid > low:
            bank[band, low:mid] = np.linspace(0.0, 1.0, mid - low, endpoint=False)
        if high > mid:
            bank[band, mid:high] = np.linspace(1.0, 0.0, high - mid, endpoint=False)
    return bank


_BANK_CACHE: dict[tuple[int, int, int], NDArray[np.float64]] = {}


def mel_distance_db(
    reference: NDArray[np.float64],
    estimate: NDArray[np.float64],
    *,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = 2048,
    hop: int = 512,
    n_bands: int = _MEL_BANDS,
) -> float:
    """RMS log error over mel bands, in dB (lower is better).

    Band energies track what a listener resolves, so this ranks models more
    faithfully than per-bin distance on spectra that neither method resolves.
    """
    key = (n_fft, sample_rate, n_bands)
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = _mel_filterbank(n_fft, sample_rate, n_bands)
    bank = _BANK_CACHE[key]

    reference, estimate = _match_lengths(reference, estimate)
    ref = stft_magnitude(reference, n_fft=n_fft, hop=hop) ** 2
    est = stft_magnitude(estimate, n_fft=n_fft, hop=hop) ** 2
    scale = float(np.sum(ref * est)) / max(float(np.sum(est * est)), 1e-30)
    ref_band = bank @ ref
    est_band = bank @ (est * scale)
    floor = max(float(np.max(ref_band)), 1e-30) * 10.0 ** (-_MEL_FLOOR_DB / 10.0)
    ref_db = 10.0 * np.log10(np.maximum(ref_band, floor))
    est_db = 10.0 * np.log10(np.maximum(est_band, floor))
    return float(np.sqrt(np.mean((ref_db - est_db) ** 2)))


def envelope_error_db(
    reference: NDArray[np.float64],
    estimate: NDArray[np.float64],
    *,
    window: int = _ENVELOPE_WINDOW,
) -> float:
    """RMS log error of the short-time energy envelope, in dB.

    This catches what a magnitude spectrogram cannot: a phase-blind model that
    smears an attack across the window scores badly here while scoring well on
    the spectrogram.
    """
    reference, estimate = _match_lengths(reference, estimate)
    trimmed = (reference.size // window) * window
    if trimmed == 0:
        return 0.0
    ref = reference[:trimmed].reshape(-1, window)
    est = estimate[:trimmed].reshape(-1, window)
    ref_rms = np.sqrt(np.mean(ref**2, axis=1))
    est_rms = np.sqrt(np.mean(est**2, axis=1))
    scale = float(np.sum(ref_rms * est_rms)) / max(float(np.sum(est_rms**2)), 1e-30)
    floor = max(float(np.max(ref_rms)), 1e-30) * 1e-4
    ref_db = 20.0 * np.log10(np.maximum(ref_rms, floor))
    est_db = 20.0 * np.log10(np.maximum(est_rms * scale, floor))
    return float(np.sqrt(np.mean((ref_db - est_db) ** 2)))


def crest_factor(signal: NDArray[np.float64]) -> float:
    """Peak-to-RMS ratio, which decides how much headroom a fixed-point render needs."""
    flat = np.asarray(signal, dtype=np.float64).ravel()
    rms = float(np.sqrt(np.mean(flat**2))) if flat.size else 0.0
    return float(np.max(np.abs(flat)) / rms) if rms > 0.0 else 0.0


@dataclass(frozen=True)
class Metrics:
    mel_distance_db: float
    log_spectral_distance_db: float
    waveform_snr_db: float
    envelope_error_db: float
    crest_factor: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate(reference: NDArray[np.float64], estimate: NDArray[np.float64]) -> Metrics:
    """Compute every reported metric for one reconstruction."""
    return Metrics(
        mel_distance_db=mel_distance_db(reference, estimate),
        log_spectral_distance_db=log_spectral_distance(reference, estimate),
        waveform_snr_db=snr_db(reference, estimate),
        envelope_error_db=envelope_error_db(reference, estimate),
        crest_factor=crest_factor(estimate),
    )
