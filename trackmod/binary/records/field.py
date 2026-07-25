import struct

import pydantic
from pydantic import BaseModel

from trackmod.schema.config import FROZEN


class Field(BaseModel):
    """One fixed field of a record: a :mod:`struct` value written at a byte offset.

    ``code`` is a ``struct`` format for a single value, such as ``"<I"`` for a 32-bit word, ``"B"`` for a
    byte, ``"b"`` for a signed byte, or ``"26s"`` for a fixed-length block the caller has already padded.
    """

    model_config = FROZEN

    name: str = pydantic.Field(min_length=1)
    offset: int = pydantic.Field(ge=0)
    code: str = pydantic.Field(min_length=1)

    @property
    def size(self) -> int:
        """How many bytes the field occupies."""
        return struct.calcsize(self.code)


class ArrayField(BaseModel):
    """A contiguous run of fixed-stride elements, such as a keyboard note map.

    ``code`` is the ``struct`` format for one element and its packed size is the stride; each value is a
    sequence unpacked into that element.
    """

    model_config = FROZEN

    name: str = pydantic.Field(min_length=1)
    offset: int = pydantic.Field(ge=0)
    count: int = pydantic.Field(ge=0)
    code: str = pydantic.Field(min_length=1)

    @property
    def stride(self) -> int:
        """How many bytes one element occupies."""
        return struct.calcsize(self.code)

    @property
    def size(self) -> int:
        """How many bytes the whole run occupies."""
        return self.stride * self.count
