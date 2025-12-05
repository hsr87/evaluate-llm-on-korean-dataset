"""Configuration module"""

from .prompts import SYSTEM_PROMPT_MULTIPLE_CHOICE
from .question_templates import (
    TYPE_1,
    TYPE_2,
    TYPE_3,
    TYPE_4,
    TYPE_MMLU_FEW_SHOT,
)

__all__ = [
    "SYSTEM_PROMPT_MULTIPLE_CHOICE",
    "TYPE_1",
    "TYPE_2",
    "TYPE_3",
    "TYPE_4",
    "TYPE_MMLU_FEW_SHOT",
]

