import re
from langchain_core.output_parsers import StrOutputParser
from core.logger import logger

class MultipleChoicesFourParser(StrOutputParser):
    """Parser for multiple choice questions with four options"""

    def parse(self, text: str) -> tuple[str, str]:
        
        logger.debug(f"🤖 Raw LLM output (4-choice): {repr(text)}")
        
        response = text.strip()
        
        # 첫 줄이 단일 문자(A-D)인 경우 바로 반환
        first_line = response.split('\n')[0].strip()
        if len(first_line) == 1 and first_line.upper() in ['A', 'B', 'C', 'D']:
            return first_line.upper(), response
        
        # ### ANSWER 섹션이 있으면 그 뒤에서 추출
        if "### ANSWER" in response:
            answer_section = response.split("### ANSWER", 1)[1]
            match = re.search(r'\b([A-D])\b', answer_section, re.IGNORECASE)
        else:
            # 기존 방식: 전체 텍스트에서 첫 번째 A-D 추출
            match = re.search(r'\b([A-D])\b', response, re.IGNORECASE)
        
        if match:
            pred = match.group(1).upper()
        else:
            pred = ""  # Wrong answer

        return pred, response


class MultipleChoicesFiveParser(StrOutputParser):
    """Parser for multiple choice questions with five options"""

    def parse(self, text: str) -> tuple[str, str]:
        
        logger.debug(f"🤖 Raw LLM output (5-choice): {repr(text)}")
        
        response = text.strip()
        
        # 첫 줄이 단일 문자(A-E)인 경우 바로 반환
        first_line = response.split('\n')[0].strip()
        if len(first_line) == 1 and first_line.upper() in ['A', 'B', 'C', 'D', 'E']:
            return first_line.upper(), response
        
        # ### ANSWER 섹션이 있으면 그 뒤에서 추출
        if "### ANSWER" in response:
            answer_section = response.split("### ANSWER", 1)[1]
            match = re.search(r'\b([A-E])\b', answer_section, re.IGNORECASE)
        else:
            # 기존 방식: 전체 텍스트에서 첫 번째 A-E 추출
            match = re.search(r'\b([A-E])\b', response, re.IGNORECASE)
        
        if match:
            pred = match.group(1).upper()
        else:
            pred = ""  # Wrong answer

        return pred, response
