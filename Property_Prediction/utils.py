import gc
import torch
import time
from litellm import completion
from .data import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams



class ModelInterface:

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        raise NotImplementedError

        
class LiteLLMModel(ModelInterface):
    """
    Modular wrapper for models supported by litellm 
    (OpenAI, Anthropic, HuggingFace, Cohere, vLLM, etc.)
    
    Usage examples:
        model = LiteLLMModel(model_name="gpt-4")
        model = LiteLLMModel(model_name="claude-3-opus-20240229")
        model = LiteLLMModel(model_name="huggingface/meta-llama/Llama-2-7b-chat-hf")
        model = LiteLLMModel(model_name="openai/custom-model", api_base="http://localhost:8000/v1")
    """
    def __init__(self, model_name: str, api_key: str = None, api_base: str = None, max_retries: int = 6, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self.kwargs = kwargs

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> list[str]:
        outputs = []
        for _ in range(n):
            text = ""
            for attempt in range(self.max_retries):
                try:
                    response = completion(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        api_key=self.api_key,
                        api_base=self.api_base,
                        **self.kwargs
                    )
                    text = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    print(f"API error on attempt {attempt+1}/{self.max_retries}: {e}")
                    time.sleep(2) # Exponential backoff can be added here
            outputs.append(text)
            time.sleep(0.5)                         
        return outputs


class HuggingFaceModel(ModelInterface):

    def __init__(self, temperature, model_name, max_new_tokens=50):
        self.temperature = temperature
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm = LLM(model=self.model_name)



    def build_chat_prompt(self,user_prompt,sys_prompt):
    
        messages = [
        {
            "role": "system",
            "content": sys_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return chat_prompt



    def generate(self, user_prompt, sys_prompt, n=1, temperature=0.7) -> list[str]:
        outputs = []

        for _ in range(n):
            prompt = self.build_chat_prompt(user_prompt, sys_prompt)

            params = SamplingParams(
                temperature=temperature,
                top_p=1.0,
                max_tokens=self.max_new_tokens,
            )

            result = self.llm.generate(prompt, params)

            for i, output in enumerate(result):
                generated_text = output.outputs[0].text
                outputs.append(generated_text.strip().split("\n")[0])

        return outputs 