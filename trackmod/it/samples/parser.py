from trackmod.binary.pcm.codec import decode_pcm
from trackmod.binary.records.values import RecordValues, read_bytes, read_int
from trackmod.binary.text import decode_name
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.loop import Loop, LoopMode
from trackmod.core.samples.sample import Sample
from trackmod.it.panning import shared_panning
from trackmod.it.spec.flags import SampleFlag, SamplePanning
from trackmod.it.spec.storage import PCM_ENCODING


def stored_depth(values: RecordValues) -> BitDepth:
    """The bit depth a sample header's flags declare its frames are stored at."""
    return BitDepth.SIXTEEN if SampleFlag(read_int(values, "flags")) & SampleFlag.SIXTEEN_BIT else BitDepth.EIGHT


def stored_frames(values: RecordValues) -> int:
    """How many bytes of waveform a sample header points at."""
    return read_int(values, "length") * stored_depth(values).bytes_per_frame


def loop_mode(flags: SampleFlag, ping_pong: SampleFlag) -> LoopMode:
    """Which direction a loop runs, given the flag that marks it as bidirectional."""
    return LoopMode.PING_PONG if ping_pong in flags else LoopMode.FORWARD


def read_loop(values: RecordValues, *, begin: str, end: str, mode: LoopMode) -> Loop | None:
    """One loop read from its pair of frame-index fields, or ``None`` when the range is empty."""
    first, last = read_int(values, begin), read_int(values, end)
    if last <= first:
        return None

    return Loop(begin=first, end=last, mode=mode)


def parse_sample(values: RecordValues, data: bytes) -> Sample:
    """Rebuild a sample from its header fields and the frames the header points at."""
    flags = SampleFlag(read_int(values, "flags"))
    depth = stored_depth(values)
    panning = read_int(values, "default_pan")
    loop = (
        read_loop(values, begin="loop_begin", end="loop_end", mode=loop_mode(flags, SampleFlag.PING_PONG_LOOP))
        if SampleFlag.LOOP in flags
        else None
    )
    sustain = (
        read_loop(
            values,
            begin="sustain_begin",
            end="sustain_end",
            mode=loop_mode(flags, SampleFlag.PING_PONG_SUSTAIN),
        )
        if SampleFlag.SUSTAIN_LOOP in flags
        else None
    )
    return Sample(
        name=decode_name(read_bytes(values, "name")),
        pcm=decode_pcm(data, depth=depth, encoding=PCM_ENCODING),
        rate=read_int(values, "c5speed"),
        depth=depth,
        volume=read_int(values, "default_volume"),
        gain=read_int(values, "global_volume"),
        panning=shared_panning(panning & ~SamplePanning.ENABLED) if panning & SamplePanning.ENABLED else None,
        loop=loop,
        sustain_loop=sustain,
    )
