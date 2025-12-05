"""Prompt templates configuration"""

import os
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT_MULTIPLE_CHOICE = """You MUST output ONLY a single uppercase letter (A, B, C, D, or E). Nothing else.

DO NOT:
- Add explanations
- Add reasoning
- Add punctuation
- Add quotes
- Add formatting
- Add any other text

ONLY output one letter. Examples:
A
B
C

WRONG (DO NOT DO THIS):
A. The answer is...
"C"
Answer: B
**D**
E\n\nExplanation:..."""

SYSTEM_PROMPT_WITH_REASONING = """You are a multiple-choice question answerer. Your task is to analyze the question carefully and select the single best answer.

INSTRUCTIONS:
1. Keep your reasoning BRIEF and CONCISE (2-3 sentences maximum)
2. After reasoning, provide your answer in the format: ### ANSWER followed by a single letter
3. Your answer must be ONLY one uppercase letter: A, B, C, D, or E

CORRECT OUTPUT FORMAT:
[Brief reasoning in 2-3 sentences]

### ANSWER
A

INCORRECT OUTPUT EXAMPLES:
- "C"
- The answer is C
- C.
- Answer: C
- (C)
- [C]
- ```C```

Remember: Keep reasoning SHORT and always use ### ANSWER format."""

def _is_reasoning_enabled():
    """Check if reasoning is enabled from environment variables"""
    return os.getenv("REASONING_ENABLED", "false").lower() == "true"

def get_system_prompt():
    """Get system prompt based on reasoning configuration"""
    if _is_reasoning_enabled():
        print("[INFO] System Prompt - Reasoning mode enabled")
        return SYSTEM_PROMPT_WITH_REASONING
    else:
        print("[INFO] System Prompt - Reasoning mode disabled")
        return SYSTEM_PROMPT_MULTIPLE_CHOICE
