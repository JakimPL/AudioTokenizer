from typing import Final

from pydantic import ConfigDict

FROZEN: Final = ConfigDict(frozen=True, extra="forbid")
FROZEN_ROOT: Final = ConfigDict(frozen=True)
