"""The one place a format name is resolved to the binding that writes it."""

from __future__ import annotations

from typing import Final, Mapping

from audiotokenizer.module.binding import Binding
from audiotokenizer.module.format import Format
from audiotokenizer.module.it import IT_BINDING
from audiotokenizer.module.xm import XM_BINDING

BINDINGS: Final[Mapping[Format, Binding]] = {Format.IT: IT_BINDING, Format.XM: XM_BINDING}


def binding_for(module_format: Format) -> Binding:
    """The binding that compiles a song into ``module_format``."""
    return BINDINGS[module_format]
