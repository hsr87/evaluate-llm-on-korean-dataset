import os
import argparse

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_community.llms.azureml_endpoint import AzureMLOnlineEndpoint
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from .phi3_formatter import CustomPhi3ContentFormatter


def str2bool(v):
    """Convert string to boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def format_timespan(seconds):
    """Format seconds into human-readable timespan."""
    hours = seconds // 3600
    minutes = (seconds - hours * 3600) // 60
    remaining_seconds = seconds - hours * 3600 - minutes * 60
    timespan = f"{hours} hours {minutes} minutes {remaining_seconds:.4f} seconds."
    return timespan


def get_prompt_template(template_type):
    """Get prompt template based on the template type."""

    if template_type == "basic":
        prompt = PromptTemplate.from_template("{question}")
    elif template_type == "chat":
        system_prompt = """You are an AI assistant who reads a given question and solves multiple choice questions.
        You don't need to write a detailed explanation of your answer in sentences. Just answer in one word like.
        
        ## Constraints
        - Just answer in one word like 'A', 'B', 'C', 'D', or 'E'.
        - Please do not answer with a sentence like 'The answer is A'.
        - Don't give multiple answers like 'A, B'.
        - Just say one choice in one word.
        """
        system_message_template = SystemMessagePromptTemplate.from_template(
            system_prompt
        )
        human_prompt = [
            {"type": "text", "text": "{question}"},
        ]
        human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

        prompt = ChatPromptTemplate.from_messages(
            [system_message_template, human_message_template]
        )
    else:
        raise ValueError(
            "Invalid 'template_type' value. Please choose from ['basic', 'chat']"
        )
    return prompt


def get_llm_client(
    model_provider, hf_model_id=None, temperature=0.01, max_tokens=256, max_retries=3
):
    """Get LLM client"""

    if model_provider == "azureopenai":
        print("Using Azure OpenAI model provider.")
        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        llm = AzureChatOpenAI(
            azure_deployment=model_name,
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    elif model_provider == "openai":
        print("Using OpenAI model provider.")
        model_name = os.getenv("OPENAI_DEPLOYMENT_NAME")
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    elif model_provider == "huggingface":
        if (
            temperature == 0.0
        ):  # in case of not supporting 0.0 for some SLM, set to 0.01
            temperature = 0.01
        model_name = hf_model_id.split("/")[-1]
        print("Using Hugging Face model provider.")
        llm = HuggingFaceEndpoint(
            repo_id=hf_model_id,
            temperature=temperature,
            max_new_tokens=max_tokens,
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
        )
    elif model_provider == "azureml":
        print("Using Azure ML endpoint as model provider.")
        model_name = os.getenv("AZURE_ML_DEPLOYMENT_NAME")

        llm = AzureMLOnlineEndpoint(
            endpoint_url=os.getenv("AZURE_ML_ENDPOINT_URL"),
            endpoint_api_type=os.getenv("AZURE_ML_ENDPOINT_TYPE"),
            endpoint_api_key=os.getenv("AZURE_ML_API_KEY"),
            content_formatter=CustomPhi3ContentFormatter(),
            model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens},
        )

    elif model_provider == "azureai":
        print("Using Azure AI Foundry endpoint as model provider.")
        model_name = os.getenv("AZURE_AI_DEPLOYMENT_NAME")

        llm = AzureAIChatCompletionsModel(
            endpoint=os.getenv("AZURE_AI_INFERENCE_ENDPOINT"),
            credential=os.getenv("AZURE_AI_INFERENCE_KEY"),
            model_name=model_name,
        )
    else:
        raise ValueError(
            "Invalid 'model_provider' value. Please choose from ['azureopenai', 'openai', 'huggingface', 'azureml', 'azureai']"
        )

    return llm, model_name
