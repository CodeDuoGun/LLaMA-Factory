import requests
from medical.config import config
import json
from typing import Generator, Optional
from medical.utils.log import logger
from medical.llm_v2.base import BaseLLMClient
import time
from openai import OpenAI


class DoubaoAIClient(BaseLLMClient):
    @property
    def llmclient(self):
        self.client = OpenAI(api_key=config.ARK_API_KEY, base_url=config.ARK_BASE_URL,timeout=(5.0, 30))
        return self.client
    
    @property
    def botclient(self):
        self.bot_client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
            api_key=config.ARK_API_KEY,
        )
        return self.bot_client

    def chat(
            self, 
            messages: list[dict], 
            stream: bool = False, 
            llm_model: str="doubao-seed-1-6-250615", 
            json_schema: dict=None, 
            max_tokens: int=600, 
            tools=None,
            temperature:float=0.6
        ) -> str:
        # url = f"{self.base_url}/v1/chat/completions"
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
            "tools": tools,
            "tool_choices": "auto"
        }
        
        if json_schema:
            payload["response_format"] = json_schema
        # TODO: @txueduo 这里处理需要思考的所有模型
        if llm_model == "doubao-seed-1-6-250615":
            payload["thinking"]  = {"type": "disabled"}

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
        prompt = [{"role": "system", "content": "网址、报告解读专家"}]
        completion = self.botclient.chat.completions.create(
            model="bot-20250924223221-txg9s",  # Bot ID
            messages=prompt + messages,
            stream=stream
        )
        return completion
        # for chunk in completion:
        #     if chunk.choices and chunk.choices[0].delta.content:
        #         print(chunk)
        #         yield chunk.choices[0].delta.content

    def image_generate(self, text:str, refer_imgs:list, edit_img:bool=False, img_nums:int=1):

        """
        @param edit_img 是否为图片编辑或者多图融合
        
        Return:
        {
            "model": "doubao-seedream-4-0-250828",
            "created": 1757322902,
            "data": [
                {
                    "url": "https://...",
                    "size": "2336x1760"
                },
            ],
            "usage": {
                "generated_images": 4,
                "output_tokens": 64240,
                "total_tokens": 64240
            }
        }
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.ARK_API_KEY}",  # 建议从环境变量里读取
        }
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        data = {
            "model": config.IMAGE_GENERATE_MODEL,
            "prompt": text,
            "size": "2K",
        }
        # TODO: 根据是图片生成还是多图融合 传递不同参数
        if refer_imgs: 
            data["image"] = refer_imgs 
        elif edit_img:
            # 图片生成，多图融合
            data.update({"image": refer_imgs, "sequential_image_generation": "disabled"})

        # 多图生成 
        else:
            data.update(
                {
                    "sequential_image_generation": "auto",
                    "sequential_image_generation_options": {
                        "max_images": img_nums
                    },
                    "stream": False,
                    "response_format": "url",
                    "watermark": True
                }
            )
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        result = resp.json()
        return result


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

# if __name__ == "__main__":
#     # Example usage
#     from medical.config import config
#     client = DoubaoAIClient(base_url=config.ARK_API_URL, api_key=config.ARK_API_KEY)
#     messages = [{"role": "user", "content": "Hello, how are you?"}]
    
#     # Streaming response
#     for chunk in client.chat(messages, stream=True):
#         print(chunk)
    
#     # Non-streaming response
#     response = client.chat(messages, stream=False)
#     print(response)