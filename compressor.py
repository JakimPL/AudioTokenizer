import os
from typing import List, Optional, Tuple
import warnings

import numpy as np
import scipy.io.wavfile as wav

from utils import int16_to_float, convert_to_mono, pad


class AudioCompressor:
    def __init__(
            self,
            unit_length: int,
            volume_resolution: int,
            channels_per_layer: int = 1,
            increase_resolution: bool = True,
            samples_per_instrument: int = 1,
            amplification: float = 1.0
    ):
        self.unit_length: int = unit_length
        self.volume_resolution: int = volume_resolution
        self.factor: int = round(np.sqrt(volume_resolution))

        self.increase_resolution: bool = increase_resolution
        self.channels_per_layer: int = channels_per_layer

        self.samples_per_instrument: int = samples_per_instrument
        self.sample_copies: int = 4 if increase_resolution else 2
        self.instrument_size: int = samples_per_instrument * self.sample_copies

        self.amplification: float = float(np.clip(amplification, 0.0, 1.0))

    def __call__(
            self,
            input_paths: List[os.PathLike],
            layers_per_signal: List[int],
            samples_per_signal: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset = 0
        samples, amplitudes, patterns = [], [], []
        for input_path, number_of_layers, number_of_samples in zip(input_paths, layers_per_signal, samples_per_signal):
            sampling_rate, signal = wav.read(input_path)
            signal = convert_to_mono(signal)

            sample_data, volume_data = self.compress_audio(signal, number_of_samples)
            number_of_samples, number_of_layers = self.get_updated_sizes(sample_data, number_of_layers)

            sample_data, volume_data, amplitude_data = self.normalize_audio(sample_data, volume_data)
            energy_data = amplitude_data * np.sum(sample_data ** 2, axis=1)

            sample_data = self.prepare_sample_data(sample_data)
            pattern_data = self.prepare_pattern_data(volume_data, number_of_samples, offset)
            pattern_data = self.compress_pattern_data(pattern_data, energy_data, number_of_layers)
            amplitude_data = amplitude_data[::self.samples_per_instrument]

            samples.append(sample_data)
            amplitudes.append(amplitude_data)
            patterns.append(pattern_data)

            offset += number_of_samples

        sample_data = np.vstack(samples)
        amplitude_data = np.hstack(amplitudes)
        pattern_data = np.vstack(patterns)

        return sample_data, amplitude_data, pattern_data

    @staticmethod
    def find_approximation(signal_matrix: np.ndarray, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        u, sigma, v = np.linalg.svd(signal_matrix.T)
        u_truncated = u[:, :samples]
        sigma_truncated = np.diag(sigma[:samples])

        alphas = v[:samples, :]
        approximation = (u_truncated @ sigma_truncated).T

        return approximation, alphas

    def divide_by_channels(self, array: np.ndarray, layers: int) -> np.ndarray:
        array = array.T.reshape(-1, self.channels_per_layer, layers).T
        return array.reshape(layers * self.channels_per_layer, -1)

    def recalculate_number_of_samples(self, real_size: int, samples: int) -> int:
        warning = False
        new_samples = samples
        if real_size < samples or self.unit_length < samples:
            new_samples = min(real_size, self.unit_length)

        new_samples = (new_samples // self.samples_per_instrument) * self.samples_per_instrument
        if new_samples == 0:
            raise ValueError("Can't find the optimal number of samples.")

        if warning:
            warnings.warn(f"Warning: Too many samples  ({samples}) requested. Limiting the number of samples to {new_samples}.")

        return new_samples

    @staticmethod
    def get_updated_sizes(sample_data: np.ndarray, number_of_layers: int) -> Tuple[int, int]:
        number_of_samples = sample_data.shape[0]
        number_of_layers = min(number_of_layers, number_of_samples)
        return number_of_samples, number_of_layers

    def compress_audio(self, signal: np.ndarray, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        length = len(signal)
        new_length = self.unit_length * ((length + self.unit_length - 1) // self.unit_length)
        padded_signal = pad(int16_to_float(signal), new_length)
        signal_matrix = padded_signal.reshape(-1, self.unit_length)
        samples = self.recalculate_number_of_samples(signal_matrix.shape[0], samples)
        return self.find_approximation(signal_matrix, samples)

    def prepare_amplitude_data(self, sample_data: np.ndarray) -> np.ndarray:
        amplitude_data = np.abs(sample_data).max(axis=1)[:, np.newaxis]
        amplitude_data_groups = np.max(amplitude_data.reshape(-1, self.samples_per_instrument), axis=1)[:, np.newaxis]
        amplitude_data = np.repeat(amplitude_data_groups, self.samples_per_instrument, axis=0)
        return amplitude_data

    def scale_amplitude_data(self, amplitude_data: np.ndarray) -> np.ndarray:
        amplitude_data = self.volume_resolution * amplitude_data / amplitude_data.max()
        return amplitude_data.reshape(-1)

    def normalize_audio(self, sample_data: np.ndarray, volume_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        volume_data = volume_data / np.abs(volume_data).max()
        volume_maximum = np.abs(volume_data).max(axis=1)[:, np.newaxis]

        sample_data = sample_data / volume_maximum
        volume_data = volume_data / volume_maximum
        sample_data = sample_data / np.abs(sample_data).max()

        amplitude_data = self.prepare_amplitude_data(sample_data)
        sample_data = sample_data / amplitude_data

        amplitude_data = self.scale_amplitude_data(amplitude_data)
        quiet_samples = np.where(amplitude_data < self.factor)
        for index in quiet_samples:
            sample_data[index] = sample_data[index] * amplitude_data[index, np.newaxis] / self.factor
            amplitude_data[index] = self.factor

        return sample_data, volume_data, amplitude_data

    def prepare_sample_data(self, sample_data: np.ndarray) -> np.ndarray:
        samples = [sample_data, -sample_data]
        if self.increase_resolution:
            factor = self.factor / self.volume_resolution
            samples += [factor * sample_data, -factor * sample_data]

        number_of_samples = sample_data.shape[0] * self.sample_copies
        sample_data = np.stack(samples).transpose((1, 0, 2))
        return sample_data.reshape(number_of_samples, -1)

    @staticmethod
    def compress_pattern_data(pattern_data: np.ndarray, energy_data: np.ndarray, max_channels: int) -> np.ndarray:
        lines = []
        max_length = 0
        for index in range(pattern_data.shape[1]):
            line = pattern_data[:, index]

            volumes = line[:, 0]
            energies = volumes * energy_data

            positive_indices = np.where(volumes > 0)[0]
            filtered_line = line[positive_indices]
            filtered_energies = energies[positive_indices]
            top_indices = np.argsort(-filtered_energies)[:max_channels]
            top_filtered_line = filtered_line[top_indices]

            max_length = max(max_length, len(top_indices))
            lines.append(top_filtered_line)

        constant_item = np.array([[-1, -1, -1]])
        final_lines = []
        for line in lines:
            constant_items = max_length - len(line)
            padding_array = np.tile(constant_item, (constant_items, 1))
            final_lines.append(np.vstack([line, padding_array]))

        return np.stack(final_lines).transpose((1, 0, 2))

    def prepare_delay_data(self, length: int, layers: int) -> np.ndarray:
        delay_data = np.expand_dims(np.arange(0, self.channels_per_layer), axis=1)
        delay_data = np.tile(delay_data, (layers, length // self.channels_per_layer))
        return delay_data

    def prepare_instruments_data(self, length: int, layers: int) -> np.ndarray:
        instruments_data = np.arange(0, layers) + 1
        instruments_data = np.tile(instruments_data, (length, 1)).T
        return self.divide_by_channels(instruments_data, layers)

    def double_pattern_data(self, instruments_data: np.ndarray, volume_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.increase_resolution:
            quiet_indices = np.where(np.abs(volume_data) < (self.factor / self.volume_resolution))
            volume_data[quiet_indices] = volume_data[quiet_indices] * self.factor
            instruments_data = 2 * (instruments_data - 1) + 1
            instruments_data[quiet_indices] = instruments_data[quiet_indices] + 1

        negative_indices = np.where(volume_data < 0)
        volume_data[negative_indices] = -volume_data[negative_indices]
        instruments_data = 2 * (instruments_data - 1) + 1
        instruments_data[negative_indices] += 1
        return instruments_data, volume_data

    def discretize_volume_data(self, instruments_data: np.ndarray, volume_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        volume_data = self.amplification * volume_data / np.abs(volume_data).max()
        volume_data = np.round(np.clip(volume_data * self.volume_resolution, -self.volume_resolution, self.volume_resolution))
        return instruments_data, volume_data.astype("int8")

    def prepare_pattern_data(self, volume_data: np.ndarray, layers, offset: int) -> np.ndarray:
        length = volume_data.shape[1]
        volume_data = self.divide_by_channels(volume_data, layers)
        instruments_data = self.prepare_instruments_data(length, layers)
        delay_data = self.prepare_delay_data(length, layers)

        instruments_data, volume_data = self.double_pattern_data(instruments_data, volume_data)
        instruments_data, volume_data = self.discretize_volume_data(instruments_data, volume_data)
        instruments_data += offset * self.instrument_size
        return np.stack([volume_data, instruments_data, delay_data]).transpose((1, 2, 0))
