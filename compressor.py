import os
from typing import Tuple, List

import numpy as np
import scipy.io.wavfile as wav

from utils import int16_to_float, convert_to_mono, pad


class AudioCompressor:
    def __init__(
            self,
            unit_length: int,
            volume_resolution: int,
            channels_per_layer: int = 1,
            pattern_compression: bool = False,
            increase_resolution: bool = True
    ):
        self.unit_length = unit_length
        self.volume_resolution = volume_resolution
        self.factor = round(np.sqrt(volume_resolution))

        self.increase_resolution = increase_resolution
        self.samples_per_instrument = 4 if increase_resolution else 2
        self.channels_per_layer = channels_per_layer

        self.pattern_compression = pattern_compression

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        y = x.copy()
        return np.maximum(y, 0, y)

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

    def compress_audio(self, signal: np.ndarray, layers: int) -> Tuple[np.ndarray, np.ndarray]:
        length = len(signal)
        new_length = self.unit_length * ((length + self.unit_length - 1) // self.unit_length)
        padded_signal = pad(int16_to_float(signal), new_length)
        signal_matrix = padded_signal.reshape(-1, self.unit_length)
        return self.find_approximation(signal_matrix, layers)

    def normalize_audio(self, sample_data: np.ndarray, volume_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        volume_data = volume_data / np.abs(volume_data).max()
        volume_maximum = np.abs(volume_data).max(axis=1)[:, np.newaxis]

        sample_data = sample_data / volume_maximum
        volume_data = volume_data / volume_maximum

        sample_data = sample_data / np.abs(sample_data).max()

        amplitude_data = np.abs(sample_data).max(axis=1)[:, np.newaxis]
        sample_data = sample_data / amplitude_data

        amplitude_data = self.volume_resolution * amplitude_data / amplitude_data.max()
        amplitude_data = amplitude_data.reshape(-1)

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

        number_of_samples = sample_data.shape[0] * self.samples_per_instrument
        sample_data = np.stack(samples).transpose((1, 0, 2))
        return sample_data.reshape(number_of_samples, -1)

    def compress_pattern_data(self, pattern_data: np.ndarray) -> np.ndarray:
        lines = []
        max_length = 0
        for index in range(pattern_data.shape[1]):
            line = pattern_data[:, index]
            volumes = line[:, 0]
            positive_indices = np.where(volumes > 0)[0]
            filtered_line = line[positive_indices]

            max_length = max(max_length, len(positive_indices))
            lines.append(filtered_line)

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
        volume_data = volume_data / np.abs(volume_data).max()
        volume_data = np.round(np.clip(volume_data * self.volume_resolution, -self.volume_resolution, self.volume_resolution))
        return instruments_data, volume_data.astype("int8")

    def prepare_pattern_data(self, volume_data: np.ndarray, layers, offset: int) -> np.ndarray:
        length = volume_data.shape[1]
        volume_data = self.divide_by_channels(volume_data, layers)
        instruments_data = self.prepare_instruments_data(length, layers)
        delay_data = self.prepare_delay_data(length, layers)

        instruments_data, volume_data = self.double_pattern_data(instruments_data, volume_data)
        instruments_data, volume_data = self.discretize_volume_data(instruments_data, volume_data)
        instruments_data += offset * self.samples_per_instrument
        return np.stack([volume_data, instruments_data, delay_data]).transpose((1, 2, 0))

    def __call__(
            self,
            input_paths: List[os.PathLike],
            layers: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset = 0
        samples, amplitudes, patterns = [], [], []
        for input_path, layers in zip(input_paths, layers):
            sampling_rate, signal = wav.read(input_path)
            signal = convert_to_mono(signal)

            sample_data, volume_data = self.compress_audio(signal, layers)
            sample_data, volume_data, amplitude_data = self.normalize_audio(sample_data, volume_data)

            sample_data = self.prepare_sample_data(sample_data)
            pattern_data = self.prepare_pattern_data(volume_data, layers, offset)

            samples.append(sample_data)
            amplitudes.append(amplitude_data)
            patterns.append(pattern_data)

            offset += layers

        sample_data = np.vstack(samples)
        amplitude_data = np.hstack(amplitudes)
        pattern_data = np.vstack(patterns)

        if self.pattern_compression:
            pattern_data = self.compress_pattern_data(pattern_data)

        return sample_data, amplitude_data, pattern_data
