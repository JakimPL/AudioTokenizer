from typing import Final

from trackmod.binary.nibble import join_nibbles
from trackmod.core.effects.effect import Effect
from trackmod.limits.guard import require_range
from trackmod.xm.effects.command import DIGITS, XMEffect, XMExtended
from trackmod.xm.spec.effects import (
    NIBBLE_PARAMETER,
    ORDER_PARAMETER,
    PANNING_PARAMETER,
    ROW_PARAMETER,
    SPEED_PARAMETER,
    TEMPO_PARAMETER,
)


def extended(sub_command: XMExtended, value: int) -> Effect:
    """An ``E`` effect carrying one sub-command and its four-bit value.

    Raises:
        ValueError: when ``value`` exceeds four bits.
    """
    return Effect(command=XMEffect.EXTENDED, parameter=join_nibbles(sub_command, value))


def decimal_parameter(value: int) -> int:
    """The byte this format reads back as the decimal number ``value``.

    FastTracker 2 inherited a pattern break whose parameter is read a digit to a nibble, so a break to
    row 16 is stored as ``0x16`` rather than ``0x10``.

    Raises:
        ValueError: when ``value`` needs more than two decimal digits.
    """
    return join_nibbles(*divmod(require_range(value, bound=ROW_PARAMETER, subject="row"), DIGITS))


class XMEffects:
    """The shared effect vocabulary as FastTracker 2 spells it.

    Speed and tempo share the one ``F`` command, split by where the parameter falls: below the tempo
    floor it sets the ticks a row lasts, and at or above it the beats per minute. That is why a tempo
    effect cannot reach the sixteen-bit ceiling the file header carries — the parameter is one byte.
    """

    def set_speed(self, ticks: int) -> Effect:
        """Set the ticks each row lasts."""
        return Effect(
            command=XMEffect.SET_SPEED,
            parameter=require_range(ticks, bound=SPEED_PARAMETER, subject="speed"),
        )

    def set_tempo(self, beats_per_minute: int) -> Effect:
        """Set the tick rate, in beats per minute."""
        return Effect(
            command=XMEffect.SET_SPEED,
            parameter=require_range(beats_per_minute, bound=TEMPO_PARAMETER, subject="tempo"),
        )

    def position_jump(self, order: int) -> Effect:
        """Continue playback at the given order-list position."""
        return Effect(
            command=XMEffect.POSITION_JUMP,
            parameter=require_range(order, bound=ORDER_PARAMETER, subject="order"),
        )

    def pattern_break(self, row: int) -> Effect:
        """End the pattern and continue at the given row of the next order."""
        return Effect(command=XMEffect.PATTERN_BREAK, parameter=decimal_parameter(row))

    def note_delay(self, ticks: int) -> Effect:
        """Delay this cell's note by the given number of ticks into the row."""
        return extended(XMExtended.NOTE_DELAY, require_range(ticks, bound=NIBBLE_PARAMETER, subject="delay"))

    def note_cut(self, ticks: int) -> Effect:
        """Silence this channel the given number of ticks into the row."""
        return extended(XMExtended.NOTE_CUT, require_range(ticks, bound=NIBBLE_PARAMETER, subject="cut"))

    def volume_slide(self, *, up: int, down: int) -> Effect:
        """Slide the channel volume by one nibble per tick, in exactly one direction.

        Raises:
            ValueError: when both directions are asked for at once, which the parameter cannot express.
        """
        if up and down:
            raise ValueError(f"a volume slide runs one way, got up {up} and down {down}")

        require_range(up, bound=NIBBLE_PARAMETER, subject="slide up")
        require_range(down, bound=NIBBLE_PARAMETER, subject="slide down")
        return Effect(command=XMEffect.VOLUME_SLIDE, parameter=join_nibbles(up, down))

    def set_panning(self, position: int) -> Effect:
        """Place the channel on the stereo field, on the shared 0..255 scale."""
        return Effect(
            command=XMEffect.SET_PANNING,
            parameter=require_range(position, bound=PANNING_PARAMETER, subject="panning"),
        )


XM_EFFECTS: Final = XMEffects()
