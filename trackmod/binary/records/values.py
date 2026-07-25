from collections.abc import Mapping, Sequence

FieldValue = int | bytes
ArrayValue = Sequence[Sequence[int]]
RecordValues = Mapping[str, FieldValue | ArrayValue]


def read_int(values: RecordValues, name: str) -> int:
    """Read one unpacked field as an integer.

    Raises:
        ValueError: when the field holds a block of bytes or a run of elements instead.
    """
    value = values[name]
    if not isinstance(value, int):
        raise ValueError(f"field {name!r} holds {type(value).__name__}, expected an integer")

    return value


def read_bytes(values: RecordValues, name: str) -> bytes:
    """Read one unpacked field as a fixed-length byte block.

    Raises:
        ValueError: when the field holds an integer or a run of elements instead.
    """
    value = values[name]
    if not isinstance(value, bytes):
        raise ValueError(f"field {name!r} holds {type(value).__name__}, expected a byte block")

    return value


def read_rows(values: RecordValues, name: str) -> ArrayValue:
    """Read one unpacked array field as its run of elements.

    Raises:
        ValueError: when the field holds a single value instead.
    """
    value = values[name]
    if isinstance(value, (int, bytes)):
        raise ValueError(f"field {name!r} holds {type(value).__name__}, expected a run of elements")

    return value
