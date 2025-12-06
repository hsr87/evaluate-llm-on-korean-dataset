"""Question prompt templates for benchmarks"""


def _get_options_string(num_choices):
    """Generate options string (e.g., 'A, B, C, D' or 'A, B, C, D, E')"""
    letters = [chr(65 + i) for i in range(num_choices)]
    return ', '.join(letters[:-1]) + f', {letters[-1]}' if num_choices > 1 else letters[0]


def _get_options_format(num_choices):
    """Generate options format string (e.g., 'A: {A}, B: {B}, C: {C}, D: {D}')"""
    return ', '.join([f'{chr(65 + i)}: {{{chr(65 + i)}}}' for i in range(num_choices)])


def get_question_template(num_choices=5, with_context=False, few_shot=False):
    """Generate question template dynamically
    
    Args:
        num_choices: Number of answer choices (4, 5, 10, etc.)
        with_context: Whether to include context/paragraph
        few_shot: Whether to include few-shot examples
    """
    options_str = _get_options_string(num_choices)
    options_format = _get_options_format(num_choices)
    
    if few_shot:
        return f"""<instruction>
You are taking a multiple choice exam. Select the single correct answer from the choices provided and respond with ONLY the letter ({options_str}). Do not include any explanation or additional text.
</instruction>

<examples>
{{FEW_SHOTS}}
</examples>

<question>{{QUESTION}}</question>

<choices>
{options_format}
</choices>

<answer>"""
    
    if with_context:
        return f"""<instruction>
You are taking a multiple choice exam. Read the context carefully, then select the single correct answer from the choices provided and respond with ONLY the letter ({options_str}). Do not include any explanation or additional text.
</instruction>

<context>
{{CONTEXT}}
</context>

<question>{{QUESTION}}</question>

<choices>
{options_format}
</choices>

<answer>"""
    
    return f"""<instruction>
You are taking a multiple choice exam. Select the single correct answer from the choices provided and respond with ONLY the letter ({options_str}). Do not include any explanation or additional text.
</instruction>

<question>{{QUESTION}}</question>

<choices>
{options_format}
</choices>

<answer>"""
