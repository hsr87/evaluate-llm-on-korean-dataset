"""HRM8K answer extraction and evaluation utilities"""
import re
import signal
import math
from contextlib import contextmanager

# Greek letter mappings (for constants)
GREEK_LETTERS = {
    'pi': math.pi, 'π': math.pi,
    'e': math.e,
    'tau': 2 * math.pi, 'τ': 2 * math.pi,
}

# Greek letter symbols (LaTeX and Unicode)
GREEK_SYMBOLS = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'omicron': 'ο',
    'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ',
    'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Alpha': 'Α', 'Beta': 'Β', 'Gamma': 'Γ', 'Delta': 'Δ', 'Epsilon': 'Ε',
    'Zeta': 'Ζ', 'Eta': 'Η', 'Theta': 'Θ', 'Iota': 'Ι', 'Kappa': 'Κ',
    'Lambda': 'Λ', 'Mu': 'Μ', 'Nu': 'Ν', 'Xi': 'Ξ', 'Omicron': 'Ο',
    'Pi': 'Π', 'Rho': 'Ρ', 'Sigma': 'Σ', 'Tau': 'Τ', 'Upsilon': 'Υ',
    'Phi': 'Φ', 'Chi': 'Χ', 'Psi': 'Ψ', 'Omega': 'Ω',
}

@contextmanager
def timeout(duration, formula):
    """Timeout context manager"""
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    """Safely evaluate mathematical expression with timeout"""
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception:
        return None

def parse_greek_expression(text):
    """Parse expression with Greek letters and return numerical value"""
    # Pattern: number followed by Greek letter (LaTeX or Unicode)
    pattern = r'(-?\d+(?:\.\d+)?)\s*(?:\\\\?(?:pi|tau|e)|[πτ])'
    match = re.search(pattern, text)
    if match:
        coef = float(match.group(1))
        # Find which Greek letter
        if 'pi' in text or 'π' in text:
            return coef * math.pi
        elif 'tau' in text or 'τ' in text:
            return coef * 2 * math.pi
        elif text.endswith('e'):
            return coef * math.e
    return None

def parse_answer_part(answer_part):
    """Parse answer from extracted text part"""
    if not answer_part:
        return None
    
    # Check for Greek letters: 8π, 8\pi, 8tau, etc.
    greek_val = parse_greek_expression(answer_part)
    if greek_val is not None:
        return greek_val
    
    # Check for LaTeX fraction: \frac{a}{b} or \\frac{a}{b}
    frac_match = re.search(r'\\\\?frac\{(-?\d+)\}\{(-?\d+)\}', answer_part)
    if frac_match:
        return float(frac_match.group(1)) / float(frac_match.group(2))
    
    # Check for plain fraction: a/b
    plain_frac = re.search(r'(-?\d+)/(-?\d+)', answer_part)
    if plain_frac:
        return float(plain_frac.group(1)) / float(plain_frac.group(2))
    
    # Regular number
    match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_part)
    if match:
        return float(match.group().replace(",", ""))
    
    return None

def extract_answer(response):
    """Extract numerical answer from model response"""
    if not response:
        return None
    
    # Convert to string if dict/list
    response_str = str(response)
    
    # Look for ### ANSWER marker first (preferred), then #### (legacy)
    # If both exist, prioritize ### ANSWER
    answer_part = None
    if "### ANSWER" in response_str:
        parts = response_str.split("### ANSWER")
        answer_part = parts[-1].strip()
        
        # If last part is empty, try second to last
        if not answer_part and len(parts) > 1:
            answer_part = parts[-2].strip()
    elif "####" in response_str:
        parts = response_str.split("####")
        answer_part = parts[-1].strip()
        
        # If last part is empty, try second to last
        if not answer_part and len(parts) > 1:
            answer_part = parts[-2].strip()
    
    if answer_part:
        result = parse_answer_part(answer_part)
        if result is not None:
            return result
    
    # Look for boxed answer
    boxed_match = re.search(r'\\\\?boxed\{([^}]+)\}', response_str)
    if boxed_match:
        result = parse_answer_part(boxed_match.group(1).strip())
        if result is not None:
            return result
    
    # Look for last Greek expression, fraction or number in response
    greek_val = parse_greek_expression(response_str)
    if greek_val is not None:
        return greek_val
    
    frac_match = re.findall(r'\\\\?frac\{(-?\d+)\}\{(-?\d+)\}', response_str)
    if frac_match:
        last_frac = frac_match[-1]
        denominator = float(last_frac[1])
        if denominator != 0:
            return float(last_frac[0]) / denominator
        else:
            return None  # 분모가 0인 경우
    
    plain_frac = re.findall(r'(-?\d+)/(-?\d+)', response_str)
    if plain_frac:
        last_frac = plain_frac[-1]
        denominator = float(last_frac[1])
        if denominator != 0:
            return float(last_frac[0]) / denominator
        else:
            return None  # 분모가 0인 경우
    
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response_str)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    
    return None
