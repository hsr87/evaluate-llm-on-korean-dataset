from langchain.schema.output_parser import StrOutputParser


class MultipleChoicesFourParser(StrOutputParser):
    """Parser for multiple choice questions with four options"""

    def parse(self, text: str) -> tuple[str, str]:
        response = text.strip().replace('"', "").replace("'", "")
        if response.startswith("A"):
            pred = "A"
        elif response.startswith("B"):
            pred = "B"
        elif response.startswith("C"):
            pred = "C"
        elif response.startswith("D"):
            pred = "D"
        else:
            pred = ""  # Wrong answer

        return pred, response


class MultipleChoicesFiveParser(StrOutputParser):
    """Parser for multiple choice questions with five options"""

    def parse(self, text: str) -> tuple[str, str]:
        response = text.strip().replace('"', "").replace("'", "")
        if response.startswith("A"):
            pred = "A"
        elif response.startswith("B"):
            pred = "B"
        elif response.startswith("C"):
            pred = "C"
        elif response.startswith("D"):
            pred = "D"
        elif response.startswith("E"):
            pred = "E"
        else:
            pred = ""  # Wrong answer

        return pred, response
