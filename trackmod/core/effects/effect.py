import pydantic
from pydantic import BaseModel

from trackmod.schema.config import FROZEN
from trackmod.spec.width import BYTE_MAX


class Effect(BaseModel):
    """One effect-column entry: a format-specific command byte and its parameter byte.

    The pair travels together because a tracker reads them together — a parameter without its command is
    meaningless. Each format's catalogue and command enumeration give the byte values their meaning.
    """

    model_config = FROZEN

    command: int = pydantic.Field(ge=0, le=BYTE_MAX)
    parameter: int = pydantic.Field(ge=0, le=BYTE_MAX)
