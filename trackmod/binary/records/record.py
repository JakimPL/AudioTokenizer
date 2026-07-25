from __future__ import annotations

import struct

import pydantic
from pydantic import BaseModel, model_validator

from trackmod.binary.records.field import ArrayField, Field
from trackmod.binary.records.values import ArrayValue, FieldValue, RecordValues
from trackmod.schema.config import FROZEN


class Record(BaseModel):
    """A fixed-size binary record described as data: a byte size plus the fields laid out within it.

    Stating the layout declaratively keeps every byte offset in one place, so the serialisers supply
    field values and never arithmetic. Offsets a record leaves undescribed are reserved and stay zero
    when packing.
    """

    model_config = FROZEN

    size: int = pydantic.Field(ge=0)
    fields: tuple[Field, ...]
    arrays: tuple[ArrayField, ...] = ()

    @model_validator(mode="after")
    def _contained(self) -> Record:
        for field in self.fields:
            if field.offset + field.size > self.size:
                raise ValueError(f"field {field.name!r} runs past the {self.size}-byte record")

        for array in self.arrays:
            if array.offset + array.size > self.size:
                raise ValueError(f"array {array.name!r} runs past the {self.size}-byte record")

        return self

    def pack(self, values: RecordValues) -> bytes:
        """Serialise ``values`` into exactly :attr:`size` bytes.

        Raises:
            KeyError: when a described field has no value.
            TypeError: when an array field is given a single value rather than a sequence.
        """
        buffer = bytearray(self.size)
        for field in self.fields:
            struct.pack_into(field.code, buffer, field.offset, values[field.name])

        for array in self.arrays:
            rows = values[array.name]
            if isinstance(rows, (int, bytes)):
                raise TypeError(f"array {array.name!r} needs a sequence of elements, got {type(rows).__name__}")

            for index, row in enumerate(rows):
                struct.pack_into(array.code, buffer, array.offset + index * array.stride, *row)

        return bytes(buffer)

    def unpack(self, data: bytes) -> dict[str, FieldValue | ArrayValue]:
        """Read every described field out of ``data``.

        Raises:
            ValueError: when ``data`` is shorter than the record.
        """
        if len(data) < self.size:
            raise ValueError(f"record needs {self.size} bytes, got {len(data)}")

        values: dict[str, FieldValue | ArrayValue] = {}
        for field in self.fields:
            (value,) = struct.unpack_from(field.code, data, field.offset)
            values[field.name] = value

        for array in self.arrays:
            values[array.name] = tuple(
                struct.unpack_from(array.code, data, array.offset + index * array.stride)
                for index in range(array.count)
            )

        return values
