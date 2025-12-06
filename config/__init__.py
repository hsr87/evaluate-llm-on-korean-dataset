"""Configuration module"""

from .prompts import get_system_prompt
from .question_templates import get_question_template

__all__ = [
    "get_system_prompt",
    "get_question_template",
]
