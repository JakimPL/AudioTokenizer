from trackmod.binary.pcm.codec import encode_pcm
from trackmod.binary.text import encode_name
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.loop import Loop, LoopMode
from trackmod.core.samples.sample import Sample
from trackmod.it.layout.sample import SAMPLE_HEADER
from trackmod.it.panning import stored_panning
from trackmod.it.spec.flags import SampleConvert, SampleFlag, SamplePanning
from trackmod.it.spec.identity import MAGIC_SAMPLE
from trackmod.it.spec.sizes import FILENAME_BYTES, NAME_BYTES
from trackmod.it.spec.storage import PCM_ENCODING


def sample_bytes(sample: Sample) -> bytes:
    """Serialise a sample's waveform as this format stores it: signed frames, no differencing."""
    return encode_pcm(sample.pcm, depth=sample.depth, encoding=PCM_ENCODING)


def loop_flags(sample: Sample) -> SampleFlag:
    """The looping switches a sample's two loops set in its header."""
    flags = SampleFlag(0)
    if sample.loop is not None:
        flags |= SampleFlag.LOOP
        if sample.loop.mode is LoopMode.PING_PONG:
            flags |= SampleFlag.PING_PONG_LOOP

    if sample.sustain_loop is not None:
        flags |= SampleFlag.SUSTAIN_LOOP
        if sample.sustain_loop.mode is LoopMode.PING_PONG:
            flags |= SampleFlag.PING_PONG_SUSTAIN

    return flags


def loop_bounds(loop: Loop | None) -> tuple[int, int]:
    """The frame range a loop stores, which is empty when the sample carries no loop."""
    return (0, 0) if loop is None else (loop.begin, loop.end)


def sample_header(sample: Sample, *, data_offset: int) -> bytes:
    """Serialise a sample header pointing at where its frames sit in the file."""
    flags = SampleFlag.DATA | loop_flags(sample)
    if sample.depth is BitDepth.SIXTEEN:
        flags |= SampleFlag.SIXTEEN_BIT

    loop_begin, loop_end = loop_bounds(sample.loop)
    sustain_begin, sustain_end = loop_bounds(sample.sustain_loop)
    panning = 0 if sample.panning is None else SamplePanning.ENABLED | stored_panning(sample.panning)
    return SAMPLE_HEADER.pack(
        {
            "magic": MAGIC_SAMPLE,
            "filename": encode_name(sample.name, FILENAME_BYTES),
            "global_volume": sample.gain,
            "flags": int(flags),
            "default_volume": sample.volume,
            "name": encode_name(sample.name, NAME_BYTES),
            "convert": int(SampleConvert.SIGNED),
            "default_pan": panning,
            "length": sample.frames,
            "loop_begin": loop_begin,
            "loop_end": loop_end,
            "c5speed": sample.rate,
            "sustain_begin": sustain_begin,
            "sustain_end": sustain_end,
            "sample_pointer": data_offset,
            "vibrato_speed": 0,
            "vibrato_depth": 0,
            "vibrato_rate": 0,
            "vibrato_waveform": 0,
        }
    )
