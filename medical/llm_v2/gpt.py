import requests
import json
from typing import Generator, Optional
from app.model.llm.base import BaseLLMClient
from openai import OpenAI


class OpenAIClient(BaseLLMClient):
    @property
    def llmclient(self):
        self.client = OpenAI()
        return self.client

    def chat(self, messages: list[dict], stream: bool = False, llm_model: str = "gpt-4o") -> str | Generator[str, None, None]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": 0.7,
            "stream": stream
        }

        if stream:
            return self._stream_chat_response(url, headers, payload)

        with requests.post(url, headers=headers, json=payload, timeout=60) as response:
            if not response.ok:
                raise Exception(f"OpenAI error: {response.text}")
            return response.json()["choices"][0]["message"]["content"]
    
    def chat_bot(self, messages:list[dict],stream: bool = False) ->Generator[str, None, None]:
        pass 

    def image_generate(self):
        pass

    def video_generate(self):
        pass


    def image_describe(self, messages:list, stream:bool=False) -> str | Generator[str, None, None]:
        pass 


    def _stream_chat_response(self, url: str, headers: dict, payload: dict) -> Generator[str, None, None]:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as response:
            if not response.ok:
                raise Exception(f"OpenAI error: {response.text}")
            yield from self._parse_stream_response(response)

    def _parse_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        try:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        line_str = line_str[6:].strip()
                        if line_str == "[DONE]":
                            break
                        try:
                            content = json.loads(line_str)["choices"][0]["delta"].get("content", "")
                            # content = eval(line_str)["choices"][0]["delta"].get("content", "")
                            yield content
                        except Exception:
                            continue
        finally:
            response.close()

# if __name__ == "__main__":
#     # Example usage
#     from medical.config import config
#     client = OpenAIClient(base_url=config.OPENIAI_BASE_URL, api_key=config.OPENIAI_API_KEY)
#     messages = [{"role": "user", "content": "Hello, how are you?"}]
    
#     # Streaming response
#     for chunk in client.chat(messages, stream=True):
#         print(chunk)
    
#     # Non-streaming response
#     response = client.chat(messages, stream=False)
#     print(response)