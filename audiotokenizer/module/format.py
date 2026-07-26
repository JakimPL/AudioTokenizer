"""The tracker formats this codec compiles into, named once so a config can carry the choice."""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class Format(StrEnum):
    """A tracker format the codec can write, named by the extension its files carry."""

    IT = "it"
    XM = "xm"
