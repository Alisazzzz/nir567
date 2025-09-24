#All stuff with llm selection is here

from typing import Optional, Dict, List, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from nir.llm import providers


class ModelManager:
    
    def __init__(self):
        self._chat_models: Dict[str, Tuple[BaseLanguageModel, str]] = {}
        self._embedding_models: Dict[str, Tuple[Embeddings, str]] = {}

    def create_chat_model(
        self,
        name: str,
        option: str,
        config: providers.ModelConfig,
        api_info: Optional[str] = None
    ) -> BaseLanguageModel:
        match option:
            case "ollama":
                model = providers.OllamaProvider().create_model(config)
            case "openai":
                model = providers.OpenAIProvider(api_info).create_model(config)
            case "hf_local":
                model = providers.HuggingFaceLocalProvider().create_model(config)
            case "hf_api":
                model = providers.HuggingFaceAPIProvider(api_info).create_model(config)
            case _:
                raise ValueError(f"Неизвестный тип LLM: {option}")
        self._chat_models[name] = (model, config.model_name)
        return model

    def get_chat_model(self, name: str) -> BaseLanguageModel:
        return self._chat_models[name][0]

    def create_embedding_model(
        self,
        name: str,
        option: str,
        model_name: str,
        api_info: Optional[str] = None
    ) -> Embeddings:
        match option:
            case "ollama":  
                model = OllamaEmbeddings(model=model_name)
            case "openai":         
                model = OpenAIEmbeddings(model=model_name, api_key=api_info)
            case "hf_local":            
                model = HuggingFaceEmbeddings(model_name=model_name)
            case _:
                raise ValueError(f"Неизвестный тип эмбеддингов: {option}")
        self._embedding_models[name] = (model, model_name)
        return model

    def get_embedding_model(self, name: str) -> Embeddings:
        return self._embedding_models[name][0]

    def list_chat_models(self) -> List[Dict[str, str]]:
        return [
            {"name": user_name, "model_name": model_name}
            for user_name, (_, model_name) in self._chat_models.items()
        ]

    def list_embedding_models(self) -> List[Dict[str, str]]:
        return [
            {"name": user_name, "model_name": model_name}
            for user_name, (_, model_name) in self._embedding_models.items()
        ]