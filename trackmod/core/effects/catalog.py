from typing import Protocol

from trackmod.core.effects.effect import Effect


class EffectCatalog(Protocol):
    """The effects both tracker formats express, so a caller can author them without picking a format.

    Every method returns the :class:`Effect` its format uses for that intent and validates the argument
    against the room the format's parameter byte leaves for it.
    """

    def set_speed(self, ticks: int) -> Effect:
        """Set the ticks each row lasts."""

    def set_tempo(self, beats_per_minute: int) -> Effect:
        """Set the tick rate, in the tracker's beats-per-minute units."""

    def position_jump(self, order: int) -> Effect:
        """Continue playback at the given order-list position."""

    def pattern_break(self, row: int) -> Effect:
        """End the pattern and continue at the given row of the next order."""

    def note_delay(self, ticks: int) -> Effect:
        """Delay this cell's note by the given number of ticks into the row."""

    def note_cut(self, ticks: int) -> Effect:
        """Silence this channel the given number of ticks into the row."""

    def volume_slide(self, *, up: int, down: int) -> Effect:
        """Slide the channel volume by one nibble per tick, in exactly one direction."""

    def set_panning(self, position: int) -> Effect:
        """Place the channel on the stereo field, on the shared 0..255 scale."""
