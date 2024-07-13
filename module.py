import os
import struct
from typing import Dict, Tuple, List

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
            loop_samples: bool = False,
            max_rows: int = 256
    ):
        self.title = title
        self.pattern_data = pattern_data
        self.sample_data = sample_data
        self.amplitude_data = amplitude_data

        self.samples = self.calculate_number_of_samples()
        self.samples_per_instrument = int(np.clip(samples_per_instrument, 1, self.samples))
        self.speed = speed

        self.channels = self.calculate_number_of_channels()
        self.rows = self.calculate_number_of_rows(max_rows)
        self.patterns = self.calculate_number_of_patterns()
        self.instruments = self.calculate_number_of_instruments()
        self.sample_length = self.sample_data.shape[-1]
        self.bpm = self.calculate_bpm()

        self.instruments_map = self.generate_instruments_map()

        self.loop_samples = loop_samples
        self.sample_size = self.sample_data.shape[-1] + 2 * self.loop_samples

    def save(self, path: os.PathLike):
        content = self.generate()
        with open(path, "wb") as f:
            f.write(content)

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


class XMModuleGenerator(ModuleGenerator):
    HEADER = "Extended Module: "
    BASE_NOTE = 0x35

    def __init__(
            self,
            title: str,
            pattern_data: np.ndarray,
            sample_data: np.ndarray,
            amplitude_data: np.ndarray,
            speed: int = 16,
            samples_per_instrument: int = 16,
            loop_samples: bool = False,
            max_rows: int = 256
    ):
        super().__init__(
            title,
            pattern_data,
            sample_data,
            amplitude_data,
            speed,
            samples_per_instrument,
            loop_samples,
            max_rows,
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

    def get_header(self) -> bytes:
        return self.HEADER.encode()

    def get_title(self) -> bytes:
        return self.pad(self.title, 20)

    def get_stripped(self) -> bytes:
        return struct.pack("B", 0x1A)

    def get_tracker_name(self) -> bytes:
        return self.pad(TRACKER_NAME, 20)

    def get_version(self) -> bytes:
        return struct.pack("BB", 0x04, 0x01)

    def get_header_size(self) -> bytes:
        return struct.pack("<4B", 0x14, 0x01, 0x00, 0x00)

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

        return b"".join(header)

    def get_pattern_header(self) -> bytes:
        return struct.pack("<4B", 0x09, 0x00, 0x00, 0x00)

    def get_packing_type(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_rows(self) -> bytes:
        return struct.pack("<H", self.rows)

    def get_pattern_size(self, size: int) -> bytes:
        return struct.pack("<H", size)

    def get_pattern_data(self, pattern: int) -> Tuple[List[bytes], int]:
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
                    lines.append(struct.pack("B", 0x80))
                    size += 1
                elif delay == 0:
                    lines.append(struct.pack("<4B", 0x87, note, instrument, volume + 16))
                    size += 4
                else:
                    lines.append(struct.pack("<5B", note, instrument, volume + 16, 0x0E, 0xD0 + delay))
                    size += 5

        return lines, size

    def generate_pattern(self, pattern: int) -> bytes:
        lines, size = self.get_pattern_data(pattern)
        return b"".join([
            self.get_pattern_header(),
            self.get_packing_type(),
            self.get_rows(),
            self.get_pattern_size(size),
            *lines
        ])

    def generate_patterns(self) -> bytes:
        return b"".join([
            self.generate_pattern(pattern)
            for pattern in range(self.patterns)
        ])

    def get_instrument_size(self) -> bytes:
        return struct.pack("<4B", 0x07, 0x01, 0x00, 0x00)

    def get_instrument_name(self, instrument: int) -> bytes:
        return self.pad(f"instrument{instrument + 1}", 22)

    def get_instrument_type(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_number_of_samples(self, instrument: int) -> bytes:
        if instrument * self.samples_per_instrument < self.samples:
            samples = self.samples_per_instrument
        else:
            samples = self.samples - instrument * self.samples_per_instrument

        return struct.pack("<H", samples)

    def get_sample_header_size(self) -> bytes:
        return struct.pack("<4B", 0x28, 0x00, 0x00, 0x00)

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
        return struct.pack(
            "<12B36B",
            0x00, 0x00, volume, 0x00,
            0x01, 0x00, volume, 0x00,
            0x02, 0x00, 0x00, 0x00,
            *(0x00, 0x00, 0x00, 0x00) * 9
        )

    def get_panning_envelope(self) -> bytes:
        return struct.pack("<48B", *(0x00, 0x00, 0x00, 0x00) * 12)

    def get_number_of_volume_envelope_points(self) -> bytes:
        return struct.pack("B", 0x02)

    def get_number_of_panning_envelope_points(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_volume_sustain_point(self) -> bytes:
        return struct.pack("B", 0x02)

    def get_volume_envelope_loop(self) -> bytes:
        return struct.pack("<H", 0)

    def get_panning_sustain_point(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_panning_envelope_loop(self) -> bytes:
        return struct.pack("<H", 0)

    def get_volume_type(self) -> bytes:
        return struct.pack("B", 0x01)

    def get_panning_type(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_vibrato(self) -> bytes:
        return struct.pack("<4B", 0x00, 0x00, 0x00, 0x00)

    def get_volume_fadeout(self) -> bytes:
        return struct.pack("<2B", 0x00, 0x04)

    def get_reserved(self) -> bytes:
        return struct.pack("<22B", *(0x00,) * 22)

    def get_sample_length(self) -> bytes:
        return struct.pack("<I", self.sample_size * 2)

    def get_sample_loop_start(self) -> bytes:
        loop_start = self.sample_length * 2 if self.loop_samples else 0
        return struct.pack("<I", loop_start)

    def get_sample_loop_end(self) -> bytes:
        loop_end = self.sample_size * 2 if self.loop_samples else 0
        return struct.pack("<I", loop_end)

    def get_sample_volume(self) -> bytes:
        return struct.pack("B", 0x40)

    def get_sample_finetune(self) -> bytes:
        return struct.pack("B", 0xE0)

    def get_sample_type(self) -> bytes:
        return struct.pack("B", 0b0010000)

    def get_sample_panning(self) -> bytes:
        return struct.pack("B", 0x7C)

    def get_sample_relative_note_number(self, sample) -> bytes:
        value = (self.BASE_NOTE - sample % self.samples_per_instrument) % 256
        return struct.pack("B", value)

    def get_sample_compression(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_sample_name(self, sample: int) -> bytes:
        return self.pad(f"sample{sample + 1}", 22)

    def get_sample_data(self, sample: int) -> bytes:
        sample_data = self.sample_data[sample] * 16383.5
        if self.loop_samples:
            constant_value = sample_data[-1]
            sample_data = np.pad(sample_data, (0, 2), mode="constant", constant_values=constant_value)

        delta_array = np.diff(sample_data, prepend=sample_data[0]).astype(np.int16)
        return delta_array.tobytes()

    def get_sample_header(self, sample: int) -> bytes:
        return b"".join([
            self.get_sample_length(),
            self.get_sample_loop_start(),
            self.get_sample_loop_end(),
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


class ITModuleGenerator(ModuleGenerator):
    HEADER = "IMPM"
    INSTRUMENT_HEADER = "IMPI"
    SAMPLE_HEADER = "IMPS"

    STRUCT_HEADER_SIZE = 192
    STRUCT_INSTRUMENT_SIZE = 554
    STRUCT_SAMPLE_SIZE = 80
    STRUCT_PATTERN_SIZE = 8

    BASE_NOTE = 0x23

    def __init__(
            self,
            title: str,
            pattern_data: np.ndarray,
            sample_data: np.ndarray,
            amplitude_data: np.ndarray,
            speed: int = 16,
            samples_per_instrument: int = 16,
            loop_samples: bool = False,
            max_rows: int = 256
    ):
        super().__init__(
            title,
            pattern_data,
            sample_data,
            amplitude_data,
            speed,
            samples_per_instrument,
            loop_samples,
            max_rows
        )

    def calculate_number_of_rows(self, max_rows: int) -> int:
        return int(np.clip(max_rows, 32, 200))

    def calculate_header_size(self) -> int:
        return self.STRUCT_HEADER_SIZE + self.patterns * 5 + self.instruments * 4 + self.samples * 4

    def calculate_instruments_offset(self) -> int:
        return self.calculate_header_size()

    def calculate_samples_offset(self) -> int:
        return self.calculate_instruments_offset() + self.instruments * self.STRUCT_INSTRUMENT_SIZE

    def calculate_patterns_offset(self) -> int:
        return self.calculate_samples_offset() + self.samples * self.STRUCT_SAMPLE_SIZE

    def calculate_sample_data_offset(self, pattern_sizes: List[int]) -> int:
        return self.calculate_patterns_offset() + sum(pattern_sizes)

    def get_header(self) -> bytes:
        return self.HEADER.encode()

    def get_title(self) -> bytes:
        return self.pad(self.title, 26)

    def get_pattern_highlight(self) -> bytes:
        return struct.pack("<BB", 0x04, 0x10)

    def get_number_of_patterns(self) -> bytes:
        return struct.pack("<H", self.patterns)

    def get_number_of_instruments(self) -> bytes:
        return struct.pack("<H", self.instruments)

    def get_number_of_samples(self) -> bytes:
        return struct.pack("<H", self.samples)

    def get_tracker_version(self) -> bytes:
        return struct.pack("<I", 0x02145131)

    def get_flags(self) -> bytes:
        return struct.pack("<I", 0x0006004D)

    def get_global_volume(self) -> bytes:
        return struct.pack("B", 0x80)

    def get_mix_volume(self) -> bytes:
        return struct.pack("B", 0x30)

    def get_speed(self) -> bytes:
        return struct.pack("B", self.speed)

    def get_bpm(self) -> bytes:
        return struct.pack("B", self.bpm)

    def get_pan_separation(self) -> bytes:
        return struct.pack("B", 0)

    def get_pitch_wheel_depth(self) -> bytes:
        return struct.pack("B", 0)

    def get_message_length(self) -> bytes:
        return struct.pack("<H", 0x00)

    def get_message_offset(self) -> bytes:
        return struct.pack("<I", 0x0000)

    def get_reserved(self) -> bytes:
        return struct.pack("<I", 0x0000)

    def get_initial_channel_pan(self) -> bytes:
        return struct.pack("<64B", *(0x20,) * 64)

    def get_initial_channel_volume(self) -> bytes:
        return struct.pack("<64B", *(0x40,) * 64)

    def get_order_table(self) -> bytes:
        return bytes(range(self.patterns))

    def get_instrument_offsets(self) -> bytes:
        base_offset = self.calculate_instruments_offset()
        offsets = [
            base_offset + instrument * self.STRUCT_INSTRUMENT_SIZE
            for instrument in range(self.instruments)
        ]

        return struct.pack(f"<{self.instruments}I", *offsets)

    def get_sample_header_offsets(self) -> bytes:
        base_offset = self.calculate_samples_offset()
        offsets = [
            base_offset + sample * self.STRUCT_SAMPLE_SIZE
            for sample in range(self.samples)
        ]

        return struct.pack(f"<{self.samples}I", *offsets)

    def get_pattern_offsets(self, pattern_sizes: List[int]) -> bytes:
        base_offset = self.calculate_patterns_offset()
        offsets = []
        total_size = 0
        for pattern in range(self.patterns):
            offset = base_offset + total_size
            offsets.append(offset)
            total_size += pattern_sizes[pattern]

        return struct.pack(f"<{self.patterns}I", *offsets)

    def generate_header(self, pattern_sizes: List[int]) -> bytes:
        header = [
            self.get_header(),
            self.get_title(),
            self.get_pattern_highlight(),
            self.get_number_of_patterns(),
            self.get_number_of_instruments(),
            self.get_number_of_samples(),
            self.get_number_of_patterns(),
            self.get_tracker_version(),
            self.get_flags(),
            self.get_global_volume(),
            self.get_mix_volume(),
            self.get_speed(),
            self.get_bpm(),
            self.get_pan_separation(),
            self.get_pitch_wheel_depth(),
            self.get_message_length(),
            self.get_message_offset(),
            self.get_reserved(),
            self.get_initial_channel_pan(),
            self.get_initial_channel_volume(),
            self.get_order_table(),
            self.get_instrument_offsets(),
            self.get_sample_header_offsets(),
            self.get_pattern_offsets(pattern_sizes)
        ]

        return b"".join(header)

    def get_instrument_header(self) -> bytes:
        return self.INSTRUMENT_HEADER.encode()

    def get_instrument_filename(self) -> bytes:
        return self.pad("", 12)

    def get_instrument_reserved(self) -> bytes:
        return struct.pack("B", 0)

    def get_instrument_new_note_action(self) -> bytes:
        return struct.pack("B", 0x01)

    def get_instrument_duplicate_check_type(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_instrument_duplicate_check_action(self) -> bytes:
        return struct.pack("B", 0x02)

    def get_instrument_fadeout(self) -> bytes:
        return struct.pack("<H", 0x00)

    def get_instrument_pitch_pan_separation(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_instrument_pitch_pan_center(self) -> bytes:
        return struct.pack("B", 0x3C)

    def get_instrument_global_volume(self, instrument: int) -> bytes:
        volume = round(2 * self.amplitude_data[instrument])
        return struct.pack("B", volume)

    def get_instrument_default_pan(self) -> bytes:
        return struct.pack("B", 0xA0)

    def get_instrument_random_volume_variation(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_instrument_random_pan_variation(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_instrument_tracker_version(self) -> bytes:
        return struct.pack("<H", 0x5131)

    def get_instrument_number_of_samples(self) -> bytes:
        return struct.pack("B", self.samples_per_instrument)

    def get_instrument_name(self, instrument: int) -> bytes:
        return self.pad(f"instrument{instrument + 1}", 26)

    def get_instrument_initial_cutoff(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_instrument_initial_resonance(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_instrument_midi_channel(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_instrument_midi_program(self) -> bytes:
        return struct.pack("B", 0xFF)

    def get_instrument_midi_bank(self) -> bytes:
        return struct.pack("<H", 0xFF)

    def get_instrument_note_sample_keyboard_table(self, instrument: int) -> bytes:
        table = []
        sample_offset = instrument * self.samples_per_instrument
        for note in range(120):
            table.append(note)
            offset = min(max(1, note - self.BASE_NOTE), self.samples_per_instrument)
            table.append(sample_offset + offset)

        return bytes(table)

    def get_volume_envelope_flags(self) -> bytes:
        return struct.pack("B", 0x01)

    def get_volume_envelope_node_count(self) -> bytes:
        return struct.pack("B", 0x03)

    def get_volume_envelope_loop_start(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_volume_envelope_loop_end(self) -> bytes:
        return struct.pack("B", 0x01)

    def get_volume_envelope_sustain_start(self) -> bytes:
        return struct.pack("B", 0x01)

    def get_volume_envelope_sustain_end(self) -> bytes:
        return struct.pack("B", 0x01)

    def get_volume_envelope_nodes(self) -> bytes:
        return struct.pack(
            "<75B",
            0x40, 0x00, 0x00,
            0x40, 0x01, 0x00,
            0x00, 0x03, 0x00,
            *(0x00, 0x00, 0x00) * 22,
        )

    def get_envelope_trailing_byte(self) -> bytes:
        return struct.pack("<1B", 0x00)

    def get_envelope_trailing_bytes(self) -> bytes:
        return struct.pack("<4B", *(0x00,) * 4)

    def get_volume_envelope(self) -> bytes:
        volume_envelope = [
            self.get_volume_envelope_flags(),
            self.get_volume_envelope_node_count(),
            self.get_volume_envelope_loop_start(),
            self.get_volume_envelope_loop_end(),
            self.get_volume_envelope_sustain_start(),
            self.get_volume_envelope_sustain_end(),
            self.get_volume_envelope_nodes(),
            self.get_envelope_trailing_byte()
        ]

        return b"".join(volume_envelope)

    def get_panning_and_pitch_envelope_flags(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_panning_and_pitch_node_count(self) -> bytes:
        return struct.pack("B", 0x02)

    def get_panning_and_pitch_loop_start(self) -> bytes:
        return struct.pack("B", 0x00)

    def get_panning_and_pitch_loop_end(self) -> bytes:
        return struct.pack("B", 0x01)

    def get_panning_and_pitch_sustain_start(self) -> bytes:
        return struct.pack("B", 0x01)

    def get_panning_and_pitch_sustain_end(self) -> bytes:
        return struct.pack("B", 0x01)

    def get_panning_and_pitch_nodes(self) -> bytes:
        return struct.pack(
            "<75B",
            0x00, 0x01, 0x00,
            0x00, 0x02, 0x00,
            *(0x00, 0x00, 0x00) * 23
        )

    def get_panning_and_pitch_envelope(self) -> bytes:
        panning_and_pitch_envelope = [
            self.get_panning_and_pitch_envelope_flags(),
            self.get_panning_and_pitch_node_count(),
            self.get_panning_and_pitch_loop_start(),
            self.get_panning_and_pitch_loop_end(),
            self.get_panning_and_pitch_sustain_start(),
            self.get_panning_and_pitch_sustain_end(),
            self.get_panning_and_pitch_nodes(),
            self.get_envelope_trailing_byte()
        ]

        return b"".join(panning_and_pitch_envelope)

    def get_instrument_envelopes(self) -> bytes:
        return b"".join([
            self.get_volume_envelope(),
            self.get_panning_and_pitch_envelope(),
            self.get_panning_and_pitch_envelope(),
            self.get_envelope_trailing_bytes()
        ])

    def get_instrument(self, instrument: int) -> bytes:
        instrument_data = [
            self.get_instrument_header(),
            self.get_instrument_filename(),
            self.get_instrument_reserved(),
            self.get_instrument_new_note_action(),
            self.get_instrument_duplicate_check_type(),
            self.get_instrument_duplicate_check_action(),
            self.get_instrument_fadeout(),
            self.get_instrument_pitch_pan_separation(),
            self.get_instrument_pitch_pan_center(),
            self.get_instrument_global_volume(instrument),
            self.get_instrument_default_pan(),
            self.get_instrument_random_volume_variation(),
            self.get_instrument_random_pan_variation(),
            self.get_instrument_tracker_version(),
            self.get_instrument_number_of_samples(),
            self.get_instrument_reserved(),
            self.get_instrument_name(instrument),
            self.get_instrument_initial_cutoff(),
            self.get_instrument_initial_resonance(),
            self.get_instrument_midi_channel(),
            self.get_instrument_midi_program(),
            self.get_instrument_midi_bank(),
            self.get_instrument_note_sample_keyboard_table(instrument),
            self.get_instrument_envelopes()
        ]

        return b"".join(instrument_data)

    def generate_instruments(self) -> bytes:
        return b"".join([
            self.get_instrument(instrument)
            for instrument in range(self.instruments)]
        )

    def get_sample_header(self) -> bytes:
        return self.SAMPLE_HEADER.encode()

    def get_sample_filename(self) -> bytes:
        return self.pad("", 12)

    def get_sample_reserved(self) -> bytes:
        return struct.pack("B", 0)

    def get_sample_global_volume(self) -> bytes:
        return struct.pack("B", 0x40)

    def get_sample_flags(self) -> bytes:
        flag = 0b00000011
        if self.loop_samples:
            flag |= 0b00010000

        return struct.pack("B", flag)

    def get_sample_default_volume(self) -> bytes:
        return struct.pack("B", 0x40)

    def get_sample_name(self, sample: int) -> bytes:
        return self.pad(f"sample{sample + 1}", 26)

    def get_sample_data_flags(self) -> bytes:
        flag = 0b00000001
        return struct.pack("B", flag)

    def get_sample_default_pan(self) -> bytes:
        return struct.pack("B", 0x20)

    def get_sample_length(self) -> bytes:
        return struct.pack("<I", self.sample_size)

    def get_sample_loop_start(self) -> bytes:
        loop_start = self.sample_length if self.loop_samples else 0
        return struct.pack("<I", loop_start)

    def get_sample_loop_end(self) -> bytes:
        loop_end = self.sample_size if self.loop_samples else 0
        return struct.pack("<I", loop_end)

    def get_sample_c5_speed(self, sample: int) -> bytes:
        sample_index = sample % self.samples_per_instrument
        frequency = round(176400 * np.power(2.0 ** (-sample_index), 1 / 12))
        return struct.pack("<I", frequency)

    def get_sample_sustain_start(self) -> bytes:
        return struct.pack("<I", 0)

    def get_sample_sustain_end(self) -> bytes:
        return struct.pack("<I", 0)

    def get_sample_pointer(self, sample: int, sample_data_offset: int) -> bytes:
        pointer = sample_data_offset + 2 * sample * self.sample_size
        return struct.pack("<I", pointer)

    def get_sample_vibrato_speed(self) -> bytes:
        return struct.pack("B", 0)

    def get_sample_vibrato_depth(self) -> bytes:
        return struct.pack("B", 0)

    def get_sample_vibrato_sweep(self) -> bytes:
        return struct.pack("B", 0)

    def get_sample_vibrato_waveform(self) -> bytes:
        return struct.pack("B", 0)

    def get_sample(self, sample: int, sample_data_offset: int) -> bytes:
        sample_data = [
            self.get_sample_header(),
            self.get_sample_filename(),
            self.get_sample_reserved(),
            self.get_sample_global_volume(),
            self.get_sample_flags(),
            self.get_sample_default_volume(),
            self.get_sample_name(sample),
            self.get_sample_data_flags(),
            self.get_sample_default_pan(),
            self.get_sample_length(),
            self.get_sample_loop_start(),
            self.get_sample_loop_end(),
            self.get_sample_c5_speed(sample),
            self.get_sample_sustain_start(),
            self.get_sample_sustain_end(),
            self.get_sample_pointer(sample, sample_data_offset),
            self.get_sample_vibrato_speed(),
            self.get_sample_vibrato_depth(),
            self.get_sample_vibrato_sweep(),
            self.get_sample_vibrato_waveform()
        ]

        return b"".join(sample_data)

    def generate_samples(self, sample_data_offset: int) -> bytes:
        return b"".join([
            self.get_sample(sample, sample_data_offset)
            for sample in range(self.samples)
        ])

    def get_pattern_data_size(self) -> bytes:
        return struct.pack("<H", 0x00)

    def get_pattern_size(self, size: int) -> bytes:
        return struct.pack("<H", size)

    def get_pattern_number_of_rows(self) -> bytes:
        return struct.pack("<H", self.rows)

    def get_pattern_reserved(self) -> bytes:
        return struct.pack("<I", 0x00)

    def get_pattern_data(self, pattern: int) -> bytes:
        offset = pattern * self.rows
        channels = []
        previous_mask = [0] * self.channels

        last_instrument = [0] * self.channels
        last_command = [0] * self.channels
        last_command_value = [0] * self.channels

        for line in range(self.rows):
            for channel in range(self.channels):
                x = offset + line
                instrument = 0x00
                delay = -1
                volume = 0
                note = 0xFF
                if x < self.pattern_data[channel].shape[0]:
                    volume, sample, delay = self.pattern_data[channel][x]
                    if sample > 0:
                        instrument, note = self.instruments_map[sample]

                note = note + 11 if instrument else 0xFF
                volume = volume if volume >= 0 else 0x00
                mask_variable = 0
                effect = 0

                if note != 0xFF:
                    mask_variable |= 0x05
                if instrument != last_instrument[channel]:
                    mask_variable |= 0x02
                    last_instrument[channel] = instrument
                if effect != last_command[channel] or (effect != 0 and last_command_value[channel] != effect & 0xFF):
                    mask_variable |= 0x08
                    last_command[channel] = effect >> 8
                    last_command_value[channel] = effect & 0xFF

                channel_marker = channel + 1
                if mask_variable != previous_mask[channel]:
                    channel_marker |= 0x80
                    previous_mask[channel] = mask_variable

                channels.append(channel_marker)
                if channel_marker & 0x80:
                    channels.append(mask_variable)

                if mask_variable & 0x01:
                    channels.append(note)
                if mask_variable & 0x02:
                    channels.append(instrument)
                if mask_variable & 0x04:
                    channels.append(volume)
                if mask_variable & 0x08:
                    channels.append(effect >> 8)
                    channels.append(effect & 0xFF)

            channels.append(0x00)

        return bytes(channels)

    def get_pattern(self, pattern: int) -> Tuple[bytes, int]:
        pattern_data = self.get_pattern_data(pattern)
        pattern_size = len(pattern_data)

        pattern_struct = [
            self.get_pattern_size(pattern_size),
            self.get_pattern_number_of_rows(),
            self.get_pattern_reserved(),
            pattern_data
        ]

        return b"".join(pattern_struct), pattern_size + self.STRUCT_PATTERN_SIZE

    def generate_patterns(self) -> Tuple[bytes, List[int]]:
        patterns = []
        pattern_sizes = []
        for pattern in range(self.patterns):
            pattern_data, size = self.get_pattern(pattern)
            patterns.append(pattern_data)
            pattern_sizes.append(size)

        return b"".join(patterns), pattern_sizes

    def generate_sample_data(self) -> bytes:
        sample_data_bytes = bytearray()

        for sample_index in range(self.samples):
            sample = np.round(self.sample_data[sample_index] * 16383.5)
            if self.loop_samples:
                constant_value = sample[-1]
                sample = np.pad(sample, (0, 2), "constant", constant_values=(constant_value, constant_value))

            sample_data_bytes.extend(sample.astype("int16").tobytes())

        return sample_data_bytes

    def generate(self) -> bytes:
        instruments = self.generate_instruments()
        patterns, pattern_sizes = self.generate_patterns()
        sample_data_offset = self.calculate_sample_data_offset(pattern_sizes)
        samples = self.generate_samples(sample_data_offset)
        header = self.generate_header(pattern_sizes)
        sample_data = self.generate_sample_data()
        return header + instruments + samples + patterns + sample_data
