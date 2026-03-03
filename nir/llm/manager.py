#All stuff with llm selection is here



#--------------------------
#---------imports----------
#--------------------------

import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from nir.llm import providers 



#--------------------------------------
#---------model manager class----------
#--------------------------------------

class ModelManager:

    CONFIG_FILE_PATH = "assets/saves/models_config.json"

    def __init__(self, config_path: Optional[str] = None):     
        self._chat_models: Dict[str, Dict[str, Any]] = {}
        self._embedding_models: Dict[str, Dict[str, Any]] = {}
        if config_path:
            self._config_path = Path(config_path)
        else:
            self._config_path = Path(self.CONFIG_FILE_PATH)
        self._load_state()


    def _serialize_config(self, config: providers.ModelConfig) -> Dict:
        return {
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "extra": config.extra
        }

    def _deserialize_config(self, config_dict: Dict) -> providers.ModelConfig:
        return providers.ModelConfig(
            model_name=config_dict["model_name"],
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens"),
            **config_dict.get("extra", {})
        )

    def _load_state(self):
        if not self._config_path.exists():
            return
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error read configuration: {e}")
            return
        for name, info in data.get("chat_models", {}).items():
            try:
                config_obj = self._deserialize_config(info["config"])
                self.create_chat_model(name=name, option=info["option"], config=config_obj, api_info=info.get("api_info"), save_to_disk=False)
            except Exception as e:
                print(f"Could not load model '{name}': {e}")

        for name, info in data.get("embedding_models", {}).items():
            try:
                self.create_embedding_model(name=name, option=info["option"], model_name=info["model_name"], api_info=info.get("api_info"), save_to_disk=False)
            except Exception as e:
                print(f"Could not load embedding '{name}': {e}")

    def _save_state(self):
        data = {
            "chat_models": {},
            "embedding_models": {}
        }
        for name, info in self._chat_models.items():
            data["chat_models"][name] = {
                "option": info["option"],
                "model_name": info["model_name"],
                "config": info["config"],
                "api_info": info.get("api_info")
            }
        for name, info in self._embedding_models.items():
            data["embedding_models"][name] = {
                "option": info["option"],
                "model_name": info["model_name"],
                "api_info": info.get("api_info")
            }
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error save configuration: {e}")

    def create_chat_model(
        self,
        name: str,
        option: str,
        config: providers.ModelConfig,
        api_info: Optional[str] = None,
        save_to_disk: bool = True
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
                raise ValueError(f"Unknown LLM type: {option}")
        config_dict = self._serialize_config(config)
        self._chat_models[name] = {
            "instance": model,
            "option": option,
            "model_name": config.model_name,
            "config": config_dict,
            "api_info": api_info
        }
        if save_to_disk:
            self._save_state()   
        return model

    def create_embedding_model(
        self,
        name: str,
        option: str,
        model_name: str,
        api_info: Optional[str] = None,
        save_to_disk: bool = True
    ) -> Embeddings:
        
        match option:
            case "ollama":  
                model = OllamaEmbeddings(model=model_name)
            case "openai":         
                model = OpenAIEmbeddings(model=model_name, api_key=api_info)
            case "hf_local":            
                model = HuggingFaceEmbeddings(model_name=model_name)
            case _:
                raise ValueError(f"Unknown embeddings type: {option}")
        
        self._embedding_models[name] = {
            "instance": model,
            "option": option,
            "model_name": model_name,
            "api_info": api_info
        }
        if save_to_disk:
            self._save_state()
        return model


    def get_chat_model(self, name: str) -> BaseLanguageModel:
        return self._chat_models[name]["instance"]
    
    def get_max_tokens_for_model(self, name: str) -> int:
        return self._chat_models[name]["config"]["max_tokens"]

    def get_embedding_model(self, name: str) -> Embeddings:
        return self._embedding_models[name]["instance"]

    def list_chat_models(self) -> List[Dict[str, Any]]:
        result = []
        for user_name, info in self._chat_models.items():
            config_data = info.get("config", {})
            result.append({
                "name": user_name,
                "model_name": info["model_name"],
                "option": info["option"],
                "temperature": config_data.get("temperature"),
                "max_tokens": config_data.get("max_tokens"),
                "has_api_key": bool(info.get("api_info"))
            })
        return result

    def list_embedding_models(self) -> List[Dict[str, Any]]:
        result = []
        for user_name, info in self._embedding_models.items():
            result.append({
                "name": user_name,
                "model_name": info["model_name"],
                "option": info["option"],
                "has_api_key": bool(info.get("api_info"))
            })
        return result
    
    def remove_model(self, name: str, model_type: str = "chat"):
        if model_type == "chat":
            if name in self._chat_models:
                del self._chat_models[name]
        elif model_type == "embedding":
            if name in self._embedding_models:
                del self._embedding_models[name]
        else:
            raise ValueError("model_type must be 'chat' or 'embedding'")
        self._save_state()