from pathlib import Path
import argparse
import json

from compressor import AudioCompressor
from module import ModuleGenerator


def infer_module_format(path: Path) -> str:
    module_format = path.suffix[1:].lower()
    if module_format not in ["xm", "it"]:
        raise ValueError(f"Unsupported module format: {module_format}. The supported formats are XM and IT.")

    return module_format


if __name__ == "__main__":
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

        LAYERS = config["layers"]
        SAMPLES = config["samples"]

        UNIT_LENGTH = config["unit_length"]
        CHANNELS_PER_LAYER = config["channels_per_layer"]

        VOLUME_RESOLUTION = config["volume_resolution"]
        INCREASE_RESOLUTION = config["increase_resolution"]
        MIN_INSTRUMENT_VOLUME_ENVELOPE = config["min_instrument_volume_envelope"]
        AMPLIFICATION = config["amplification"]

        SAMPLES_PER_INSTRUMENT = config["samples_per_instrument"]
        MAX_ROWS = config["max_rows"]

    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument("--inputs", "-i", nargs="+", required=True, help="Input paths")
    parser.add_argument("--layers", "-l", nargs="+", type=int, required=False, help="Layer sizes")
    parser.add_argument("--samples", "-s", nargs="+", type=int, required=False, help="Sample sizes")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path for the module file")
    parser.add_argument("--title", "-t", type=str, default="Tokenizer", help="Title of the module (default: Tokenizer)")
    parser.add_argument("--unit", "-u", type=int, default=UNIT_LENGTH, help="The size of a token.")
    parser.add_argument("--amplify", "-a", type=float, default=AMPLIFICATION, help="Amplify the volume data.")

    args = parser.parse_args()
    input_paths = args.inputs
    layers = args.layers or [LAYERS] * len(input_paths)
    samples = args.samples or [SAMPLES] * len(input_paths)
    output_path = Path(args.output)
    title = args.title
    unit_length = args.unit
    amplification = args.amplify

    module_format = infer_module_format(output_path)
    module_generator_class = ModuleGenerator.get_module_generator_class(module_format)

    audio_compressor = AudioCompressor(
        unit_length=unit_length,
        volume_resolution=VOLUME_RESOLUTION,
        channels_per_layer=CHANNELS_PER_LAYER,
        increase_resolution=INCREASE_RESOLUTION,
        min_instrument_volume_envelope=MIN_INSTRUMENT_VOLUME_ENVELOPE,
        samples_per_instrument=SAMPLES_PER_INSTRUMENT,
        amplification=amplification
    )

    sample_data, amplitude_data, pattern_data = audio_compressor(
        input_paths=input_paths,
        layers_per_signal=layers,
        samples_per_signal=samples
    )

    samples_per_instrument = SAMPLES_PER_INSTRUMENT * (4 if INCREASE_RESOLUTION else 2)
    mg = module_generator_class(
        title=title,
        pattern_data=pattern_data,
        sample_data=sample_data,
        amplitude_data=amplitude_data,
        samples_per_instrument=samples_per_instrument,
        speed=CHANNELS_PER_LAYER,
        max_rows=MAX_ROWS
    )

    mg.save(output_path)
