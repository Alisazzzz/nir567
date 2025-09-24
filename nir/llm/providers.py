#All providers for different llms

from typing import Optional
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseLanguageModel

from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class ModelConfig:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = kwargs


class BaseModelProvider(ABC):
    @abstractmethod
    def create_model(self, config: ModelConfig) -> BaseLanguageModel:
        pass


class OllamaProvider(BaseModelProvider):
    def create_model(self, config: ModelConfig) -> BaseLanguageModel:
        return OllamaLLM(
            model=config.model_name,
            temperature=config.temperature,
            num_predict=config.max_tokens,
        )


class OpenAIProvider(BaseModelProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_model(self, config: ModelConfig) -> BaseLanguageModel:
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=self.api_key,
        )


class HuggingFaceLocalProvider(BaseModelProvider):
    def create_model(self, config: ModelConfig) -> BaseLanguageModel:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=0 if torch.cuda.is_available() else -1,
            max_length=config.max_tokens or 100,
        )
        return HuggingFacePipeline(pipeline=hf_pipeline)


class HuggingFaceAPIProvider(BaseModelProvider):
    def __init__(self, api_token: str):
        self.api_token = api_token

    def create_model(self, config: ModelConfig) -> BaseLanguageModel:
        return HuggingFaceEndpoint(
            repo_id=config.model_name,
            temperature=config.temperature,
            max_new_tokens=config.max_tokens or 100,
            huggingfacehub_api_token=self.api_token,
        )