"""HRM8K answer extraction and evaluation utilities"""
import re
import signal
from contextlib import contextmanager

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

def extract_answer(response):
    """Extract numerical answer from model response"""
    if not response:
        return None
    
    # Look for #### marker (GSM8K format)
    if "####" in response:
        answer_part = response.split("####")[-1].strip()
        match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_part)
        if match:
            return float(match.group().replace(",", ""))
    
    # Look for boxed answer
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        answer_str = boxed_match.group(1).strip()
        match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_str)
        if match:
            return float(match.group().replace(",", ""))
    
    # Look for last number in response
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    
    return None
