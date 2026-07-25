from enum import StrEnum, unique


@unique
class Severity(StrEnum):
    """How badly a value breaks a format.

    A structural value the record layout cannot hold is refused at every compliance level; a compliance
    value the layout holds but the original tracker ignores is refused only under canonical compliance.
    """

    STRUCTURAL = "structural"
    COMPLIANCE = "compliance"
