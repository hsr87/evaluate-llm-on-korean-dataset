import os
import argparse
import boto3
import time
from functools import wraps
from typing import Any, Dict, List, Optional

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_community.llms.azureml_endpoint import AzureMLOnlineEndpoint
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from .phi3_formatter import CustomPhi3ContentFormatter
from config.prompts import get_system_prompt


class ThrottledChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse with throttling protection"""
    
    def __init__(self, wait_time: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, 'wait_time', wait_time)
        object.__setattr__(self, 'last_request_time', 0)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        # Add throttling delay
        current_time = time.time()
        last_request_time = getattr(self, 'last_request_time', 0)
        time_since_last = current_time - last_request_time
        wait_time = getattr(self, 'wait_time', 10.0)
        
        if time_since_last < wait_time:
            sleep_time = wait_time - time_since_last
            time.sleep(sleep_time)
        
        object.__setattr__(self, 'last_request_time', time.time())
        return super()._generate(messages, stop, run_manager, **kwargs)
    

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


def get_prompt_template(template_type, model_provider=None):
    """Get prompt template based on the template type.
    
    Returns:
        tuple: (prompt_template, system_prompt_text or None)
    """

    system_prompt_text = get_system_prompt()
    
    if template_type == "basic":
        prompt = PromptTemplate.from_template("{question}")
    elif template_type == "chat":
        system_message_template = SystemMessagePromptTemplate.from_template(
            f"System:\n\n{system_prompt_text}"
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
    return prompt, system_prompt_text


def get_provider_name(model_provider):
    """Get human-readable provider name"""
    provider_names = {
        "azureopenai": "Azure OpenAI",
        "openai": "OpenAI",
        "bedrock": "AWS Bedrock",
        "huggingface": "Hugging Face",
        "azureml": "Azure ML endpoint",
        "azureai": "Azure AI Foundry endpoint"
    }
    return provider_names.get(model_provider, model_provider)


def get_llm_client(
    model_provider, hf_model_id=None, temperature=0.01, max_tokens=256, max_retries=3, wait_time=1.0, system_prompt=None
):
    """Get LLM client"""

    if model_provider == "azureopenai":

        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        reasoning_effort = os.getenv("REASONING_EFFORT")
        
        # GPT-5.1 계열 모델들 (reasoning_effort 지원)
        if model_name in ["gpt-51", "gpt-51-chat"]:
            model_kwargs = {}
            if reasoning_effort:
                model_kwargs["reasoning_effort"] = reasoning_effort
            
            llm = AzureChatOpenAI(
                azure_deployment=model_name,
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                max_retries=max_retries,
                model_kwargs=model_kwargs,
            )
        # GPT-5 계열 모델들 (gpt-5-chat 추가)
        elif model_name in ["gpt-5-mini", "gpt-5-nano", "gpt-5-chat"]:
            llm = AzureChatOpenAI(
                azure_deployment=model_name,
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                max_retries=max_retries,
            )
        elif model_name in ["gpt-oss-120b", "gpt-oss-20b"]:
            llm = AzureChatOpenAI(
                azure_deployment=model_name,
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                max_retries=max_retries,
                temperature=temperature,
                stop=["\n"]
            )
        else:
            llm = AzureChatOpenAI(
                azure_deployment=model_name,
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
    elif model_provider == "openai":

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

        llm = HuggingFaceEndpoint(
            repo_id=hf_model_id,
            temperature=temperature,
            max_new_tokens=max_tokens,
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
        )
    elif model_provider == "azureml":

        model_name = os.getenv("AZURE_ML_DEPLOYMENT_NAME")

        llm = AzureMLOnlineEndpoint(
            endpoint_url=os.getenv("AZURE_ML_ENDPOINT_URL"),
            endpoint_api_type=os.getenv("AZURE_ML_ENDPOINT_TYPE"),
            endpoint_api_key=os.getenv("AZURE_ML_API_KEY"),
            content_formatter=CustomPhi3ContentFormatter(),
            model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens},
        )

    elif model_provider == "azureai":

        model_name = os.getenv("AZURE_AI_DEPLOYMENT_NAME")

        llm = AzureAIChatCompletionsModel(
            endpoint=os.getenv("AZURE_AI_INFERENCE_ENDPOINT"),
            credential=os.getenv("AZURE_AI_INFERENCE_KEY"),
            model_name=model_name,
        )
    elif model_provider == "bedrock":

        model_name = os.getenv("BEDROCK_MODEL_ID")
        
        # Build additional_model_request_fields
        additional_fields = {}
        
        # Reasoning config
        reasoning_enabled = os.getenv("REASONING_ENABLED", "false").lower() == "true"
        if reasoning_enabled:
            reasoning_effort = os.getenv("REASONING_EFFORT", "high")
            additional_fields["reasoningConfig"] = {
                "type": "enabled",
                "maxReasoningEffort": reasoning_effort
            }
        
        # Prepare system prompt
        system_messages = None
        if system_prompt:
            system_messages = [system_prompt] if isinstance(system_prompt, str) else system_prompt
        
        # When reasoning is enabled with high effort, temperature and max_tokens must be unset
        bedrock_kwargs = {
            "model": model_name,
            "region_name": os.getenv("AWS_REGION"),
            "wait_time": wait_time,
            "system": system_messages,
            "additional_model_request_fields": additional_fields if additional_fields else None
        }
        
        if not (reasoning_enabled and reasoning_effort == "high"):
            bedrock_kwargs["temperature"] = temperature
            bedrock_kwargs["max_tokens"] = max_tokens
        
        llm = ThrottledChatBedrockConverse(**bedrock_kwargs)
    else:
        raise ValueError(
            "Invalid 'model_provider' value. Please choose from ['azureopenai', 'openai', 'huggingface', 'azureml', 'azureai', 'bedrock']"
        )

    return llm, model_name
