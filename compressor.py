import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.io.wavfile as wav

from utils import int16_to_float, convert_to_mono, pad


class AudioCompressor:
    def __init__(
            self,
            unit_length: int,
            channels_per_layer: int = 1,
            volume_resolution: int = 64,
            increase_volume_resolution: bool = True,
            min_instrument_volume_envelope: int = 1,
            remove_sample_slope: bool = True,
            samples_per_instrument: int = 1,
            amplification: float = 1.0,
            return_reconstruction: bool = False
    ):
        self.unit_length: int = unit_length
        self.volume_resolution: int = volume_resolution
        self.factor: int = round(np.sqrt(volume_resolution))

        self.return_reconstruction: bool = return_reconstruction

        self.increase_volume_resolution: bool = increase_volume_resolution
        self.min_instrument_volume_envelope: int = min_instrument_volume_envelope
        self.channels_per_layer: int = channels_per_layer

        self.remove_sample_slope: bool = remove_sample_slope
        self.samples_per_instrument: int = samples_per_instrument
        self.sample_copies: int = 4 if increase_volume_resolution else 2
        self.instrument_size: int = samples_per_instrument * self.sample_copies

        self.amplification: float = float(np.clip(amplification, 0.0, 1.0))

    def __call__(
            self,
            input_paths: List[os.PathLike],
            layers_per_signal: List[int],
            samples_per_signal: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        offset = 0
        samples, amplitudes, patterns = [], [], []
        for input_path, number_of_layers, number_of_samples in zip(input_paths, layers_per_signal, samples_per_signal):
            sampling_rate, signal = wav.read(input_path)
            signal = convert_to_mono(signal)

            sample_data, volume_data = self.compress_audio(signal, number_of_samples)
            number_of_samples, number_of_layers = self.get_updated_sizes(sample_data, number_of_layers)

            instruments = number_of_samples // self.samples_per_instrument
            if instruments >= 256:
                raise ValueError(f"The number of instruments ({instruments}) is more than 255 allowed.")

            sample_data, volume_data, amplitude_data = self.normalize_audio(sample_data, volume_data)

            energy_data = self.prepare_energy_data(sample_data, amplitude_data)
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

        reconstruction = None
        if self.return_reconstruction:
            if self.channels_per_layer > 1:
                warnings.warn("Warning: Reconstruction is not supported for the number of channels per layer greater than 1.")
            else:
                reconstruction = self.reconstruct_audio(sample_data, amplitude_data, pattern_data)

        return sample_data, amplitude_data, pattern_data, reconstruction

    def reconstruct_audio(self, sample_data: np.ndarray, amplitude_data: np.ndarray, pattern_data: np.ndarray) -> np.ndarray:
        amplitude_data = amplitude_data.repeat(self.instrument_size) / self.volume_resolution
        sample_data = (sample_data * 16383.5).astype("int16") * amplitude_data[:, np.newaxis]

        samples = pattern_data[:, :, 1] - 1
        volumes = pattern_data[:, :, 0]
        mask = samples < 0

        reconstruction = np.zeros((samples.shape[0], samples.shape[1], self.unit_length), dtype="float")
        valid_mask = ~mask
        reconstruction[valid_mask] = (
                (volumes[valid_mask] / self.volume_resolution)[:, np.newaxis] *
                sample_data[samples[valid_mask]]
        )

        channel_reconstructions = reconstruction.reshape(reconstruction.shape[0], -1)
        return np.round(channel_reconstructions.sum(axis=0)) / 16383.5

    @staticmethod
    def find_approximation(signal_matrix: np.ndarray, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        u, sigma, v = np.linalg.svd(signal_matrix.T)
        u_truncated = u[:, :samples]
        sigma_truncated = np.diag(sigma[:samples])

        alphas = v[:samples, :]
        approximation = (u_truncated @ sigma_truncated).T

        return approximation, alphas

    def divide_by_channels(self, array: np.ndarray, layers: int, fill_value: Union[int, float] = 0) -> np.ndarray:
        length = array.shape[-1]
        new_length = int(np.ceil(length / self.channels_per_layer)) * self.channels_per_layer
        new_array = np.full((array.shape[0], new_length), fill_value)
        new_array[:, :length] = array

        array = new_array.T.reshape(-1, self.channels_per_layer, layers).T
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

    def get_updated_sizes(self, sample_data: np.ndarray, number_of_layers: int) -> Tuple[int, int]:
        number_of_samples = sample_data.shape[0]
        number_of_layers = min(120, number_of_layers, number_of_samples * self.channels_per_layer)
        return number_of_samples, number_of_layers

    def remove_slope(self, signal_matrix: np.ndarray) -> np.ndarray:
        x = np.arange(self.unit_length)
        intercept = signal_matrix[:, 0]
        slope = (signal_matrix[:, -1] - intercept) / (self.unit_length - 1)
        trend = slope * x[:, np.newaxis] + intercept
        return signal_matrix - trend.T

    def reduce_signal(self, signal_matrix: np.ndarray, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.remove_sample_slope:
            signal_matrix = self.remove_slope(signal_matrix)

        return self.find_approximation(signal_matrix, samples)

    def compress_audio(self, signal: np.ndarray, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        length = len(signal)
        new_length = self.unit_length * ((length + self.unit_length - 1) // self.unit_length)
        padded_signal = pad(int16_to_float(signal), new_length)
        signal_matrix = padded_signal.reshape(-1, self.unit_length)
        samples = self.recalculate_number_of_samples(signal_matrix.shape[0], samples)
        return self.reduce_signal(signal_matrix, samples)

    def prepare_amplitude_data(self, sample_data: np.ndarray) -> np.ndarray:
        amplitude_data = np.abs(sample_data).max(axis=1)[:, np.newaxis]
        amplitude_data_groups = np.max(amplitude_data.reshape(-1, self.samples_per_instrument), axis=1)[:, np.newaxis]
        amplitude_data = np.repeat(amplitude_data_groups, self.samples_per_instrument, axis=0)
        return amplitude_data

    def scale_amplitude_data(self, amplitude_data: np.ndarray) -> np.ndarray:
        amplitude_data = self.volume_resolution * amplitude_data / amplitude_data.max()
        return np.ceil(amplitude_data.reshape(-1))

    def normalize_sample_data(self, sample_data: np.ndarray, amplitude_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        min_instrument_volume_envelope = np.clip(self.min_instrument_volume_envelope, 1, self.volume_resolution)
        quiet_samples = np.where(amplitude_data < min_instrument_volume_envelope)[0]

        for index in quiet_samples:
            sample_data[index] *= amplitude_data[index, np.newaxis] / min_instrument_volume_envelope
            amplitude_data[index] = min_instrument_volume_envelope

        return sample_data, amplitude_data

    def normalize_audio(self, sample_data: np.ndarray, volume_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        volume_data = volume_data / np.abs(volume_data).max()
        volume_maximum = np.abs(volume_data).max(axis=1)[:, np.newaxis]

        sample_data = sample_data / volume_maximum
        volume_data = volume_data / volume_maximum
        sample_data = sample_data / np.abs(sample_data).max()

        amplitude_data = self.prepare_amplitude_data(sample_data)
        sample_data = sample_data / amplitude_data

        amplitude_data = self.scale_amplitude_data(amplitude_data)
        sample_data, amplitude_data = self.normalize_sample_data(sample_data, amplitude_data)

        return sample_data, volume_data, amplitude_data

    @staticmethod
    def prepare_energy_data(sample_data: np.ndarray, amplitude_data: np.ndarray) -> np.ndarray:
        sample_data_energy = np.sum((sample_data - sample_data.mean(axis=1)[:, np.newaxis]) ** 2, axis=1)
        return amplitude_data * sample_data_energy

    def prepare_sample_data(self, sample_data: np.ndarray) -> np.ndarray:
        samples = [sample_data, -sample_data]
        if self.increase_volume_resolution:
            factor = self.factor / self.volume_resolution
            samples += [factor * sample_data, -factor * sample_data]

        number_of_samples = sample_data.shape[0] * self.sample_copies
        sample_data = np.stack(samples).transpose((1, 0, 2))
        return sample_data.reshape(number_of_samples, -1)

    def compress_pattern_data(self, pattern_data: np.ndarray, energy_data: np.ndarray, max_channels: int) -> np.ndarray:
        lines = []
        max_length = 0
        for index in range(pattern_data.shape[1]):
            line = pattern_data[:, index]

            volumes = line[:, 0]
            energies = volumes * np.repeat(energy_data, self.channels_per_layer)
            energies = energies.reshape(-1, self.channels_per_layer).T
            volumes = volumes.reshape(-1, self.channels_per_layer).T

            length = 0
            top_filtered_line = []
            for subline in range(self.channels_per_layer):
                positive_indices = np.where(volumes[subline] > 0)[0]
                filtered_line = line[positive_indices * self.channels_per_layer + subline]
                filtered_energies = energies[subline][positive_indices]
                top_indices = np.argsort(-filtered_energies)[:max_channels]
                top_filtered_line.extend(filtered_line[top_indices])
                length += len(top_indices)

            max_length = max(max_length, length)
            lines.append(np.array(top_filtered_line))

        constant_item = np.array([[-1, -1, -1]])
        final_lines = []
        for line in lines:
            constant_items = max_length - len(line)
            padding_array = np.tile(constant_item, (constant_items, 1))
            final_line = np.vstack([line, padding_array]) if len(line) else padding_array
            final_lines.append(final_line)

        return np.stack(final_lines).transpose((1, 0, 2))

    def prepare_delay_data(self, length: int, layers: int) -> np.ndarray:
        delay_data = np.expand_dims(np.arange(0, self.channels_per_layer), axis=1)
        delay_data = np.tile(delay_data, (layers, length // self.channels_per_layer))
        return delay_data

    def prepare_instruments_data(self, length: int, layers: int) -> np.ndarray:
        instruments_data = np.arange(0, layers) + 1
        instruments_data = np.tile(instruments_data, (length, 1)).T
        return self.divide_by_channels(instruments_data, layers, fill_value=0)

    def double_pattern_data(self, instruments_data: np.ndarray, volume_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.increase_volume_resolution:
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
        volume_data = self.divide_by_channels(volume_data, layers, fill_value=0.0)
        instruments_data = self.prepare_instruments_data(length, layers)
        delay_data = self.prepare_delay_data(length, layers)

        instruments_data, volume_data = self.double_pattern_data(instruments_data, volume_data)
        instruments_data, volume_data = self.discretize_volume_data(instruments_data, volume_data)
        instruments_data += offset * self.instrument_size
        return np.stack([volume_data, instruments_data, delay_data]).transpose((1, 2, 0))
