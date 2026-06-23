import requests
from abc import ABC, abstractmethod
from typing import Generator, Optional


class BaseLLMClient(ABC):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def chat(self, messages: list[dict], stream: bool = False, llm_model:str="") -> str | Generator[str, None, None]:
        """
        统一接口，子类根据API格式实现
        :param messages: [{"role": "user", "content": "hello"}]
        :param stream: 是否流式
        """
        pass

    @abstractmethod
    def chat_bot(self, messages:list[dict],stream: bool = False) ->Generator[str, None, None]:
        pass 
    
    @abstractmethod
    def image_generate(self):
        pass
    
    @abstractmethod
    def video_generate(self):
        pass

    @abstractmethod
    def image_describe(self, messages:list, stream:bool=False) -> str | Generator[str, None, None]:
        pass 
