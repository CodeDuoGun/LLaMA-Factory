import requests
from medical.config import config
import json
from typing import Generator, Optional
from app.utils.log import logger
from app.model.llm_v2.base import BaseLLMClient
import time
from openai import OpenAI


class BaichuanAIClient(BaseLLMClient):
    @property
    def llmclient(self):
        self.client = OpenAI(api_key=config.BAICHUAN_API_KEY, base_url=config.BAICHUAN_API_URL,timeout=(5.0, 30))
        return self.client

    def chat(
            self, 
            messages: list[dict], 
            stream: bool = False, 
            llm_model: str="Baichuan-M2-Plus", 
            json_schema: dict=None, 
            max_tokens: int=600, 
            tools=None,
            temperature:float=0.6
        ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.debug(f"***model:{llm_model}")
        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            # "tools": tools,
            # "tool_choices": "auto"
        }
        
        if json_schema:
            payload["response_format"] = {"type":"json_object"}# json_schema

        t0 = time.time()
        if stream:
            return self._stream_chat_response(headers, payload)

        with requests.post(self.base_url, headers=headers, json=payload, timeout=60) as response:
            if not response.ok:
                raise Exception(f"OpenAI error: {response.text}")
            if tools:
                return response.json()
            logger.debug(f"Request took {time.time() - t0} seconds")
            return response.json()["choices"][0]["message"]["content"]
    
    def chat_bot(self, messages, stream=True):
        pass

    def image_generate(self, text:str, refer_imgs:list, edit_img:bool=False, img_nums:int=1):

        """

        """
        pass

    def video_generate(self):
        pass


    def image_describe(self, messages:list, stream:bool=False) -> str | Generator[str, None, None]:
        """
        "image_url": {
            # 需要注意：传入Base64编码前需要增加前缀 data:image/{图片格式};base64,{Base64编码}：
            # PNG图片："url":  f"data:image/png;base64,{base64_image}"
            # JPEG图片："url":  f"data:image/jpeg;base64,{base64_image}"
            # WEBP图片："url":  f"data:image/webp;base64,{base64_image}"
                "url":  f"data:image/<IMAGE_FORMAT>;base64,{base64_image}"
            }, 
        """
        response = self.llmclient.chat.completions.create(
            model=config.IMAGE_DESCRIBE_MODEL,
            messages=messages,
            stream=stream
        )
        return response
        

    def _stream_chat_response(self, headers: dict, payload: dict) -> Generator[str, None, None]:
        with requests.post(self.base_url, headers=headers, json=payload, stream=True, timeout=60) as response:
            if not response.ok:
                raise Exception(f"OpenAI error: {response.text}")
            yield from self._parse_stream_response(response)

    def _parse_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        try:
            for line in response.iter_lines():
                # print(f"line: {line}")
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        line_str = line_str[6:].strip()
                        # print(f"line_str: {line_str}")
                        if line_str == "[DONE]":
                            break
                        try:
                            content = json.loads(line_str)["choices"][0]["delta"].get("content", "")
                            yield content
                        except Exception as e:
                            logger.error(f"Error parsing error: {e}")
                            continue
        finally:
            response.close()
