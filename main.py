import argparse
import json

from compressor import AudioCompressor
from module import ModuleGenerator

if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

        UNIT_LENGTH = config["unit_length"]
        LAYERS = config["layers"]
        CHANNELS_PER_LAYER = config["channels_per_layer"]
        VOLUME_RESOLUTION = config["volume_resolution"]
        INCREASE_RESOLUTION = config["increase_resolution"]

        MAX_ROWS = config["max_rows"]
        PATTERN_COMPRESSION = config["pattern_compression"]

        SAMPLES_PER_INSTRUMENT = 4 if INCREASE_RESOLUTION else 2

    parser = argparse.ArgumentParser(description='Process audio files.')
    parser.add_argument('--inputs', '-i', nargs='+', required=True, help='Input paths')
    parser.add_argument('--layers', '-l', nargs='+', type=int, required=False, help='Layer sizes')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output path for the module file')
    parser.add_argument('--title', '-t', type=str, default='Tokenizer', help='Title of the module (default: Tokenizer)')
    parser.add_argument('--unit', '-u', type=int, default=UNIT_LENGTH, help='The size of a token.')

    args = parser.parse_args()
    input_paths = args.inputs
    layers = args.layers or [LAYERS] * len(input_paths)
    output_path = args.output
    title = args.title

    audio_compressor = AudioCompressor(
        unit_length=UNIT_LENGTH,
        volume_resolution=VOLUME_RESOLUTION,
        channels_per_layer=CHANNELS_PER_LAYER,
        increase_resolution=INCREASE_RESOLUTION,
        pattern_compression=PATTERN_COMPRESSION
    )

    sample_data, amplitude_data, pattern_data = audio_compressor(input_paths, layers)

    mg = ModuleGenerator(
        title,
        pattern_data=pattern_data,
        sample_data=sample_data,
        amplitude_data=amplitude_data,
        samples_per_instrument=SAMPLES_PER_INSTRUMENT,
        speed=CHANNELS_PER_LAYER,
        max_rows=MAX_ROWS
    )

    mg.save(output_path)
