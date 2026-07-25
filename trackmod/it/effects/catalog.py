from typing import Final

from trackmod.binary.nibble import join_nibbles
from trackmod.core.effects.effect import Effect
from trackmod.it.effects.command import ITEffect, ITExtended
from trackmod.it.spec.effects import (
    NIBBLE_PARAMETER,
    ORDER_PARAMETER,
    PANNING_PARAMETER,
    ROW_PARAMETER,
    SPEED_PARAMETER,
    TEMPO_PARAMETER,
)
from trackmod.limits.guard import require_range


def extended(sub_command: ITExtended, value: int) -> Effect:
    """An ``S`` effect carrying one sub-command and its four-bit value.

    Raises:
        ValueError: when ``value`` exceeds four bits.
    """
    return Effect(command=ITEffect.EXTENDED, parameter=join_nibbles(sub_command, value))


class ITEffects:
    """The shared effect vocabulary as Impulse Tracker spells it."""

    def set_speed(self, ticks: int) -> Effect:
        """Set the ticks each row lasts."""
        return Effect(
            command=ITEffect.SET_SPEED,
            parameter=require_range(ticks, bound=SPEED_PARAMETER, subject="speed"),
        )

    def set_tempo(self, beats_per_minute: int) -> Effect:
        """Set the tick rate, in beats per minute."""
        return Effect(
            command=ITEffect.SET_TEMPO,
            parameter=require_range(beats_per_minute, bound=TEMPO_PARAMETER, subject="tempo"),
        )

    def position_jump(self, order: int) -> Effect:
        """Continue playback at the given order-list position."""
        return Effect(
            command=ITEffect.POSITION_JUMP,
            parameter=require_range(order, bound=ORDER_PARAMETER, subject="order"),
        )

    def pattern_break(self, row: int) -> Effect:
        """End the pattern and continue at the given row of the next order."""
        return Effect(
            command=ITEffect.PATTERN_BREAK,
            parameter=require_range(row, bound=ROW_PARAMETER, subject="row"),
        )

    def note_delay(self, ticks: int) -> Effect:
        """Delay this cell's note by the given number of ticks into the row."""
        return extended(ITExtended.NOTE_DELAY, require_range(ticks, bound=NIBBLE_PARAMETER, subject="delay"))

    def note_cut(self, ticks: int) -> Effect:
        """Silence this channel the given number of ticks into the row."""
        return extended(ITExtended.NOTE_CUT, require_range(ticks, bound=NIBBLE_PARAMETER, subject="cut"))

    def volume_slide(self, *, up: int, down: int) -> Effect:
        """Slide the channel volume by one nibble per tick, in exactly one direction.

        Raises:
            ValueError: when both directions are asked for at once, which the parameter cannot express.
        """
        if up and down:
            raise ValueError(f"a volume slide runs one way, got up {up} and down {down}")

        require_range(up, bound=NIBBLE_PARAMETER, subject="slide up")
        require_range(down, bound=NIBBLE_PARAMETER, subject="slide down")
        return Effect(command=ITEffect.VOLUME_SLIDE, parameter=join_nibbles(up, down))

    def set_panning(self, position: int) -> Effect:
        """Place the channel on the stereo field, on the shared 0..255 scale."""
        return Effect(
            command=ITEffect.SET_PANNING,
            parameter=require_range(position, bound=PANNING_PARAMETER, subject="panning"),
        )


IT_EFFECTS: Final = ITEffects()
