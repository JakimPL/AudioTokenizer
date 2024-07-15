import argparse
import json
from pathlib import Path

from compressor import AudioCompressor
from module import ModuleGenerator
from utils import infer_module_format, save_sample

if __name__ == "__main__":
    with open("config.json", "r") as config_file:
        config = json.load(config_file)

        LAYERS = config["compressor"]["layers"]
        SAMPLES = config["compressor"]["samples"]
        UNIT_LENGTH = config["compressor"]["unit_length"]
        REMOVE_SAMPLE_SLOPE = config["compressor"]["remove_sample_slope"]
        RETURN_RECONSTRUCTION = config["compressor"]["return_reconstruction"]

        VOLUME_RESOLUTION = config["module"]["volume_resolution"]
        INCREASE_VOLUME_RESOLUTION = config["module"]["increase_volume_resolution"]
        MIN_INSTRUMENT_VOLUME_ENVELOPE = config["module"]["min_instrument_volume_envelope"]
        AMPLIFICATION = config["module"]["amplification"]

        CHANNELS_PER_LAYER = config["module"]["channels_per_layer"]
        SAMPLES_PER_INSTRUMENT = config["module"]["samples_per_instrument"]
        LOOP_SAMPLES = config["module"]["loop_samples"]
        MAX_ROWS = config["module"]["max_rows"]

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
        return_reconstruction=RETURN_RECONSTRUCTION,
        channels_per_layer=CHANNELS_PER_LAYER,
        volume_resolution=VOLUME_RESOLUTION,
        increase_volume_resolution=INCREASE_VOLUME_RESOLUTION,
        min_instrument_volume_envelope=MIN_INSTRUMENT_VOLUME_ENVELOPE,
        remove_sample_slope=REMOVE_SAMPLE_SLOPE,
        samples_per_instrument=SAMPLES_PER_INSTRUMENT,
        amplification=amplification
    )

    sample_data, amplitude_data, pattern_data, reconstruction = audio_compressor(
        input_paths=input_paths,
        layers_per_signal=layers,
        samples_per_signal=samples
    )

    if RETURN_RECONSTRUCTION and reconstruction is not None:
        reconstruction_path = output_path.with_suffix(".wav")
        save_sample(reconstruction, reconstruction_path)

    samples_per_instrument = SAMPLES_PER_INSTRUMENT * (4 if INCREASE_VOLUME_RESOLUTION else 2)
    mg = module_generator_class(
        title=title,
        pattern_data=pattern_data,
        sample_data=sample_data,
        amplitude_data=amplitude_data,
        samples_per_instrument=samples_per_instrument,
        loop_samples=LOOP_SAMPLES,
        speed=CHANNELS_PER_LAYER,
        max_rows=MAX_ROWS
    )

    mg.save(output_path)
