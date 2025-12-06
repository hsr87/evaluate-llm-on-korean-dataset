import re
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.logger import logger

class ReasoningStrOutputParser(StrOutputParser):
    """Custom parser for models with reasoning_content (e.g., Nova)"""
    
    def parse(self, output):
        if isinstance(output, AIMessage):
            if isinstance(output.content, list):
                # Extract only 'text' type content, ignore 'reasoning_content'
                text_parts = [item['text'] for item in output.content if item.get('type') == 'text']
                return ' '.join(text_parts) if text_parts else ''
            return str(output.content)
        return str(output)

class BaseMultipleChoiceParser(ReasoningStrOutputParser):
    """Base parser for multiple choice questions"""
    
    def __init__(self, choices):
        super().__init__()
        self._choices = choices
        self._pattern = f"([{''.join(choices)}])"
    
    def parse(self, text: str) -> tuple[str, str]:
        if os.getenv("IS_DEBUG", "false").lower() == "true":
            with open("debug_qna.log", "a", encoding="utf-8") as f:
                f.write(f"[PARSER-{len(self._choices)}] Raw input: {repr(text)}, type: {type(text)}\n")
        
        if not text or not text.strip():
            if os.getenv("IS_DEBUG", "false").lower() == "true":
                with open("debug_qna.log", "a", encoding="utf-8") as f:
                    f.write(f"[PARSER-{len(self._choices)}] ⚠️ EMPTY RESPONSE DETECTED\n")
            return "", "[EMPTY_RESPONSE]"
        
        response = text.strip()
        
        # 1. 첫 줄이 단일 문자인 경우
        first_line = response.split('\n')[0].strip()
        if len(first_line) == 1 and first_line.upper() in self._choices:
            return first_line.upper(), response
        
        # 2. ### ANSWER 섹션
        if "### ANSWER" in response:
            answer_section = response.split("### ANSWER", 1)[1].strip()
            first_char = answer_section.split()[0] if answer_section.split() else answer_section[:1]
            if first_char.upper() in self._choices:
                return first_char.upper(), response
        
        # 3. "Answer: A" 형식
        answer_match = re.search(r'(?:answer|답변|정답)[\s:：]*([A-J])', response, re.IGNORECASE)
        if answer_match and answer_match.group(1).upper() in self._choices:
            return answer_match.group(1).upper(), response
        
        # 4. "(A)" 또는 "[A]" 형식
        bracket_match = re.search(r'[\(\[]([A-J])[\)\]]', response, re.IGNORECASE)
        if bracket_match and bracket_match.group(1).upper() in self._choices:
            return bracket_match.group(1).upper(), response
        
        # 5. "A)" 또는 "A." 형식 (줄 시작)
        option_match = re.search(r'^([A-J])[\.\)]', response, re.IGNORECASE | re.MULTILINE)
        if option_match and option_match.group(1).upper() in self._choices:
            return option_match.group(1).upper(), response
        
        # 6. XML 태그 <answer>A</answer>
        xml_match = re.search(r'<answer>([A-J])</answer>', response, re.IGNORECASE)
        if xml_match and xml_match.group(1).upper() in self._choices:
            return xml_match.group(1).upper(), response
        
        # 7. 마지막 줄이 단일 문자인 경우
        last_line = response.split('\n')[-1].strip()
        if len(last_line) == 1 and last_line.upper() in self._choices:
            return last_line.upper(), response
        
        # 8. 일반 패턴 매칭 (fallback)
        match = re.search(self._pattern, response, re.IGNORECASE)
        
        if match:
            pred = match.group(1).upper()
        else:
            if os.getenv("IS_DEBUG", "false").lower() == "true":
                with open("debug_qna.log", "a", encoding="utf-8") as f:
                    f.write(f"[PARSER-{len(self._choices)}] ⚠️ NO VALID ANSWER ({', '.join(self._choices)}) FOUND: {response[:200]}\n")
            pred = ""
        
        return pred, response


class MultipleChoicesFourParser(BaseMultipleChoiceParser):
    """Parser for multiple choice questions with four options"""
    def __init__(self):
        super().__init__(['A', 'B', 'C', 'D'])


class MultipleChoicesFiveParser(BaseMultipleChoiceParser):
    """Parser for multiple choice questions with five options"""
    def __init__(self):
        super().__init__(['A', 'B', 'C', 'D', 'E'])


class MultipleChoicesTenParser(BaseMultipleChoiceParser):
    """Parser for multiple choice questions with ten options (A-J)"""
    def __init__(self):
        super().__init__(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
