from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from trackmod.spec.grid import GRID_DIMENSIONS, GRID_DTYPE, WAVEFORM_DIMENSIONS, WAVEFORM_DTYPE


@dataclass(frozen=True)
class Shaped:
    """Validates a model field as a numpy array of one dtype and one number of dimensions.

    Arrays are the one part of the model that is not a plain value, and describing them as annotation
    metadata keeps them inside the schema every other field is checked by, rather than beside it.
    """

    dtype: type[np.generic]
    dimensions: int

    def __get_pydantic_core_schema__(self, source: type[Any], handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """The validator pydantic runs for a field annotated with this shape."""
        return core_schema.no_info_plain_validator_function(self.validate)

    def validate(self, value: object) -> NDArray[Any]:
        """Coerce ``value`` to this shape's dtype and confirm its rank.

        Raises:
            ValueError: when the value has the wrong number of dimensions.
        """
        array = np.asarray(value, dtype=self.dtype)
        if array.ndim != self.dimensions:
            raise ValueError(f"expected {self.dimensions} dimensions, got {array.ndim}")

        return array


Waveform = Annotated[NDArray[np.float64], Shaped(WAVEFORM_DTYPE, WAVEFORM_DIMENSIONS)]
Grid = Annotated[NDArray[np.int16], Shaped(GRID_DTYPE, GRID_DIMENSIONS)]
