import os
import struct
from typing import Dict, Tuple

import numpy as np

TRACKER_NAME = "Stage Magician"
C_NOTE = 0x19


class ModuleGenerator:
    def __init__(
            self,
            title: str,
            pattern_data: np.ndarray,
            sample_data: np.ndarray,
            amplitude_data: np.ndarray,
            speed: int = 16,
            samples_per_instrument: int = 16,
            max_rows: int = 256
    ):
        self.title = title
        self.pattern_data = pattern_data
        self.sample_data = sample_data
        self.amplitude_data = amplitude_data

        self.samples = self.calculate_number_of_samples()
        self.samples_per_instrument = np.clip(samples_per_instrument, 1, self.samples)
        self.speed = speed

        self.channels = self.calculate_number_of_channels()
        self.rows = self.calculate_number_of_rows(max_rows)
        self.patterns = self.calculate_number_of_patterns()
        self.instruments = self.calculate_number_of_instruments()
        self.sample_length = self.sample_data.shape[-1]
        self.bpm = self.calculate_bpm()

        self.instruments_map = self.generate_instruments_map()

    @classmethod
    def get_module_generator_class(cls, module_format: str):
        if module_format == "xm":
            return XMModuleGenerator

        if module_format == "it":
            return ITModuleGenerator

        raise ValueError(f"Unsupported module format: {module_format}")

    def calculate_number_of_samples(self) -> int:
        return self.sample_data.shape[0]

    def calculate_number_of_channels(self) -> int:
        return self.pattern_data.shape[0]

    def calculate_number_of_rows(self, max_rows: int) -> int:
        raise NotImplementedError

    def calculate_number_of_patterns(self) -> int:
        length = self.pattern_data.shape[1]
        return int(np.ceil(length / self.rows))

    def calculate_number_of_instruments(self) -> int:
        instruments = self.samples / self.samples_per_instrument
        return int(np.ceil(instruments))

    def calculate_bpm(self) -> int:
        bpm = round(110250 / self.sample_length)
        return int(np.clip(bpm, 32, 511))

    def generate_instruments_map(self) -> Dict[int, Tuple[int, int]]:
        instrument = 1
        note = C_NOTE
        i = 0

        instruments_map = {}
        for sample in range(1, self.samples + 1):
            instruments_map[sample] = (instrument, note)
            note += 1
            i += 1
            if i == self.samples_per_instrument:
                i = 0
                instrument += 1
                note = C_NOTE

        return instruments_map

    def pad(self, string: str, length: int, pad_value: int = 0x00) -> bytes:
        string = string[:length].encode()
        return string + bytes([pad_value] * (length - len(string)))

    def get_title(self) -> bytes:
        raise NotImplementedError

    def get_tracker_name(self) -> bytes:
        raise NotImplementedError

    def generate(self) -> bytes:
        raise NotImplementedError

    def save(self, path: os.PathLike):
        with open(path, "wb") as f:
            content = self.generate()
            f.write(content)


class XMModuleGenerator(ModuleGenerator):
    HEADER = "Extended Module: "

    def __init__(
            self,
            title: str,
            pattern_data: np.ndarray,
            sample_data: np.ndarray,
            amplitude_data: np.ndarray,
            speed: int = 16,
            samples_per_instrument: int = 16,
            max_rows: int = 256
    ):
        super().__init__(
            title,
            pattern_data,
            sample_data,
            amplitude_data,
            speed,
            samples_per_instrument,
            max_rows
        )

    def calculate_number_of_samples(self) -> int:
        return self.sample_data.shape[0]

    def calculate_number_of_channels(self) -> int:
        return self.pattern_data.shape[0]

    def calculate_number_of_rows(self, max_rows: int) -> int:
        rows = int(65535 / (self.channels * 5))
        return max(1, min(rows, max_rows))

    def calculate_number_of_patterns(self) -> int:
        length = self.pattern_data.shape[1]
        return int(np.ceil(length / self.rows))

    def calculate_number_of_instruments(self) -> int:
        instruments = self.samples / self.samples_per_instrument
        return int(np.ceil(instruments))

    def pad(self, string: str, length: int, pad_value: int = 0x00) -> bytes:
        string = string[:length].encode()
        return string + bytes([pad_value] * (length - len(string)))

    def get_relative_note_number(self, note: int, base: int = C_NOTE) -> int:
        return base + note

    def get_header(self) -> bytes:
        return self.HEADER.encode()

    def get_title(self) -> bytes:
        return self.pad(self.title, 20)

    def get_stripped(self) -> bytes:
        return struct.pack("<B", 0x1A)

    def get_tracker_name(self) -> bytes:
        return self.pad(TRACKER_NAME, 20)

    def get_version(self) -> bytes:
        return struct.pack('BB', 0x04, 0x01)

    def get_header_size(self) -> bytes:
        return struct.pack('<4B', 0x14, 0x01, 0x00, 0x00)

    def get_song_length(self) -> bytes:
        return struct.pack("<H", self.patterns)

    def get_restart_position(self) -> bytes:
        return struct.pack("<H", 0)

    def get_number_of_channels(self) -> bytes:
        return struct.pack("<H", self.channels)

    def get_number_of_patterns(self) -> bytes:
        return struct.pack("<H", self.patterns)

    def get_number_of_instruments(self) -> bytes:
        return struct.pack("<H", self.instruments)

    def get_flags(self) -> bytes:
        return struct.pack("<H", 1)

    def get_speed(self) -> bytes:
        return struct.pack("<H", self.speed)

    def get_bpm(self) -> bytes:
        return struct.pack("<H", self.bpm)

    def get_order_table(self) -> bytes:
        order_table_size = 256
        table = []
        for i in range(order_table_size):
            if i < self.patterns:
                table.append(i)
            else:
                table.append(0)

        return bytes(table)

    def generate_header(self) -> bytes:
        header = [
            self.get_header(),
            self.get_title(),
            self.get_stripped(),
            self.get_tracker_name(),
            self.get_version(),
            self.get_header_size(),
            self.get_song_length(),
            self.get_restart_position(),
            self.get_number_of_channels(),
            self.get_number_of_patterns(),
            self.get_number_of_instruments(),
            self.get_flags(),
            self.get_speed(),
            self.get_bpm(),
            self.get_order_table()
        ]

        return b''.join(header)

    def get_pattern_header(self) -> bytes:
        return struct.pack('<4B', 0x09, 0x00, 0x00, 0x00)

    def get_packing_type(self) -> bytes:
        return struct.pack('<B', 0x00)

    def get_rows(self) -> bytes:
        return struct.pack("<H", self.rows)

    def get_pattern_size(self, size: int) -> bytes:
        return struct.pack("<H", size)

    def generate_pattern(self, pattern: int) -> bytes:
        size = 0
        offset = pattern * self.rows
        lines = []
        for line in range(self.rows):
            for channel in range(self.channels):
                x = offset + line
                note = C_NOTE
                instrument = -1
                delay = -1
                volume = -1
                if x < self.pattern_data[channel].shape[0]:
                    volume, sample, delay = self.pattern_data[channel][x]
                    if sample > 0:
                        instrument, note = self.instruments_map[sample]

                if delay == -1:
                    lines.append(bytes([0x80]))
                    size += 1
                elif delay == 0:
                    lines.append(bytes([0x87, note, instrument, volume + 16]))
                    size += 4
                else:
                    lines.append(bytes([note, instrument, volume + 16, 0x0E, 0xD0 + delay]))
                    size += 5

        return b"".join([
            self.get_pattern_header(),
            self.get_packing_type(),
            self.get_rows(),
            self.get_pattern_size(size),
            *lines
        ])

    def generate_patterns(self) -> bytes:
        return b''.join([
            self.generate_pattern(pattern)
            for pattern in range(self.patterns)
        ])

    def get_instrument_size(self) -> bytes:
        return struct.pack('<4B', 0x07, 0x01, 0x00, 0x00)

    def get_instrument_name(self, instrument: int) -> bytes:
        return self.pad(f"instrument{instrument + 1}", 22)

    def get_instrument_type(self) -> bytes:
        return struct.pack('<B', 0x00)

    def get_number_of_samples(self, instrument: int) -> bytes:
        if instrument * self.samples_per_instrument < self.samples:
            samples = self.samples_per_instrument
        else:
            samples = self.samples - instrument * self.samples_per_instrument

        return struct.pack("<H", samples)

    def get_sample_header_size(self) -> bytes:
        return struct.pack('<4B', 0x28, 0x00, 0x00, 0x00)

    def get_keymap_assignment(self) -> bytes:
        assignment = []
        for i in range(96):
            value = i - C_NOTE
            if value < self.samples_per_instrument:
                assignment.append(max(value + 1, 0))
            else:
                assignment.append(0)

        return bytes(assignment)

    def get_volume_envelope(self, volume: int = 0x40) -> bytes:
        return struct.pack('<12B36B',
                           0x00, 0x00, volume, 0x00,
                           0x01, 0x00, volume, 0x00,
                           0x02, 0x00, 0x00, 0x00,
                           *(0x00, 0x00, 0x00, 0x00) * 9)

    def get_panning_envelope(self) -> bytes:
        return struct.pack('<48B', *(0x00, 0x00, 0x00, 0x00) * 12)

    def get_number_of_volume_envelope_points(self) -> bytes:
        return struct.pack('<B', 0x02)

    def get_number_of_panning_envelope_points(self) -> bytes:
        return struct.pack('<B', 0x00)

    def get_volume_sustain_point(self) -> bytes:
        return struct.pack('<B', 0x02)

    def get_volume_envelope_loop(self) -> bytes:
        return struct.pack('<H', 0)

    def get_panning_sustain_point(self) -> bytes:
        return struct.pack('<B', 0x00)

    def get_panning_envelope_loop(self) -> bytes:
        return struct.pack('<H', 0)

    def get_volume_type(self) -> bytes:
        return struct.pack('<B', 0x01)

    def get_panning_type(self) -> bytes:
        return struct.pack('<B', 0x00)

    def get_vibrato(self) -> bytes:
        return struct.pack('<4B', 0x00, 0x00, 0x00, 0x00)

    def get_volume_fadeout(self) -> bytes:
        return struct.pack('<2B', 0x00, 0x04)

    def get_reserved(self) -> bytes:
        return struct.pack('<22B', *(0x00,) * 22)

    def get_sample_length(self) -> bytes:
        return struct.pack('<I', self.sample_length * 2)

    def get_sample_loop(self) -> bytes:
        return struct.pack('<4B', 0x00, 0x00, 0x00, 0x00)

    def get_sample_volume(self) -> bytes:
        return struct.pack('<B', 0x40)

    def get_sample_finetune(self) -> bytes:
        return struct.pack('<B', 0xE0)

    def get_sample_type(self) -> bytes:
        return struct.pack('<B', 0b0010000)

    def get_sample_panning(self) -> bytes:
        return struct.pack('<B', 0x7C)

    def get_sample_relative_note_number(self, sample) -> bytes:
        base_value = 0x35
        value = (base_value - sample % self.samples_per_instrument) % 256
        return struct.pack('<B', value)

    def get_sample_compression(self) -> bytes:
        return struct.pack('<B', 0x00)

    def get_sample_name(self, sample: int) -> bytes:
        return self.pad(f"sample{sample + 1}", 22)

    def get_sample_data(self, sample: int) -> bytes:
        sample_data = self.sample_data[sample] * 16383.5
        delta_array = np.diff(sample_data, prepend=sample_data[0]).astype(np.int16)
        return delta_array.tobytes()

    def get_sample_header(self, sample: int) -> bytes:
        return b"".join([
            self.get_sample_length(),
            self.get_sample_loop(),
            self.get_sample_loop(),
            self.get_sample_volume(),
            self.get_sample_finetune(),
            self.get_sample_type(),
            self.get_sample_panning(),
            self.get_sample_relative_note_number(sample),
            self.get_sample_compression(),
            self.get_sample_name(sample)
        ])

    def get_instrument(self, instrument: int) -> bytes:
        volume = round(self.amplitude_data[instrument])
        instrument_data = [
            self.get_instrument_size(),
            self.get_instrument_name(instrument),
            self.get_instrument_type(),
            self.get_number_of_samples(instrument),
            self.get_sample_header_size(),
            self.get_keymap_assignment(),
            self.get_volume_envelope(volume),
            self.get_panning_envelope(),
            self.get_number_of_volume_envelope_points(),
            self.get_number_of_panning_envelope_points(),
            self.get_volume_sustain_point(),
            self.get_volume_envelope_loop(),
            self.get_panning_sustain_point(),
            self.get_panning_envelope_loop(),
            self.get_volume_type(),
            self.get_panning_type(),
            self.get_vibrato(),
            self.get_volume_fadeout(),
            self.get_reserved()
        ]

        sample_headers = []
        sample_data = []
        for sample in range(self.samples_per_instrument):
            offset = instrument * self.samples_per_instrument
            if offset + sample >= self.samples:
                break

            sample_headers.append(self.get_sample_header(offset + sample))
            sample_data.append(self.get_sample_data(offset + sample))

        return b"".join(instrument_data + sample_headers + sample_data)

    def generate_instruments(self) -> bytes:
        return b"".join([
            self.get_instrument(instrument)
            for instrument in range(self.instruments)]
        )

    def generate(self) -> bytes:
        header = self.generate_header()
        patterns = self.generate_patterns()
        instruments = self.generate_instruments()

        return header + patterns + instruments

    def save(self, path: os.PathLike):
        with open(path, "wb") as f:
            content = self.generate()
            f.write(content)


class ITModuleGenerator(ModuleGenerator):
    HEADER = "IMPM"

    def __init__(
            self,
            title: str,
            pattern_data: np.ndarray,
            sample_data: np.ndarray,
            amplitude_data: np.ndarray,
            speed: int = 16,
            samples_per_instrument: int = 16,
            max_rows: int = 256
    ):
        super().__init__(
            title,
            pattern_data,
            sample_data,
            amplitude_data,
            speed,
            samples_per_instrument,
            max_rows
        )
