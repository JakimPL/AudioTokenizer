from typing import Final

from trackmod.binary.pcm.codec import encode_pcm
from trackmod.binary.text import encode_name
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.loop import LoopMode
from trackmod.core.samples.sample import Sample
from trackmod.xm.layout.sample import SAMPLE_HEADER
from trackmod.xm.spec.defaults import DEFAULT_PANNING
from trackmod.xm.spec.flags import LoopType, SampleFlag
from trackmod.xm.spec.sizes import NAME_BYTES
from trackmod.xm.spec.storage import PCM_ENCODING
from trackmod.xm.tuning import Tuning

LOOP_TYPES: Final[dict[LoopMode, LoopType]] = {
    LoopMode.FORWARD: LoopType.FORWARD,
    LoopMode.PING_PONG: LoopType.PING_PONG,
}


def sample_bytes(sample: Sample) -> bytes:
    """Serialise a sample's waveform as this format stores it: successive differences, not amplitudes."""
    return encode_pcm(sample.pcm, depth=sample.depth, encoding=PCM_ENCODING)


def loop_type(sample: Sample) -> LoopType:
    """The loop mode a sample's header declares.

    Raises:
        ValueError: when the sample carries a sustain loop, which this format has no field for.
    """
    if sample.sustain_loop is not None:
        raise ValueError(f"sample {sample.name!r} carries a sustain loop, which this format cannot store")

    if sample.loop is None:
        return LoopType.NONE

    return LOOP_TYPES[sample.loop.mode]


def sample_header(sample: Sample, *, tuning: Tuning) -> bytes:
    """Serialise a sample header, whose lengths count bytes rather than frames."""
    stride = sample.depth.bytes_per_frame
    flags = int(loop_type(sample))
    if sample.depth is BitDepth.SIXTEEN:
        flags |= SampleFlag.SIXTEEN_BIT

    loop = sample.loop
    return SAMPLE_HEADER.pack(
        {
            "length": sample.frames * stride,
            "loop_begin": 0 if loop is None else loop.begin * stride,
            "loop_length": 0 if loop is None else loop.frames * stride,
            "volume": sample.volume,
            "finetune": tuning.finetune,
            "type": flags,
            "panning": DEFAULT_PANNING if sample.panning is None else sample.panning,
            "relative_note": tuning.relative_note,
            "reserved": 0,
            "name": encode_name(sample.name, NAME_BYTES),
        }
    )
