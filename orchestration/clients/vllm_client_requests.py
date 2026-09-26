import requests
import logging
from dataclasses import dataclass
from typing import Optional, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    text: str
    success: bool
    error_message: Optional[str] = None

class RequestsVLLMClient:
    """
    Synchronous vLLM Client using requests.
    Designed for use with ThreadPoolExecutor for concurrency.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8005/v1", api_key: str = "EMPTY", model_name: str = "casperhansen/llama-3.3-70b-instruct-awq", session: Any = None):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.completion_url = f"{base_url}/completions"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        # Session ignored in this implementation to ensure thread safety/freshness, or can use if passed
        self.session = session

    def generate(self, prompt: str, system_prompt: str = None, config: Any = None) -> GenerationResult:
        """
        Sync generation with exact Llama 3 prompt formatting parity.
        """
        try:
            # 1. Dynamic Prompt Formatting
            if "mistral" in self.model_name.lower():
                sys_part = f"{system_prompt}\n\n" if system_prompt else ""
                full_prompt = f"<s>[INST] {sys_part}{prompt} [/INST]"
            elif "gemma" in self.model_name.lower():
                sys_part = f"{system_prompt}\n\n" if system_prompt else ""
                full_prompt = f"<start_of_turn>user\n{sys_part}{prompt}<end_of_turn>\n<start_of_turn>model\n"
            else:
                # Llama 3 Template
                if system_prompt:
                     full_prompt = (
                         f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                         f"{system_prompt}<|eot_id|>"
                         f"<|start_header_id|>user<|end_header_id|>\n\n"
                         f"{prompt}<|eot_id|>"
                         f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                     )
                else:
                     full_prompt = (
                         f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                         f"{prompt}<|eot_id|>"
                         f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                     )

            # 2. Config Extraction
            temperature = config.temperature if config and hasattr(config, 'temperature') else 0.7
            max_tokens = config.max_tokens if config and hasattr(config, 'max_tokens') else 500
            
            # 3. Payload Construction
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": ["<|eot_id|>", "<|end_of_text|>", "User:", "System:", "Note:", "``` \n", "```\n\n"]
            }

            # 4. Sync Request
            # Use session if provided (and thread-safe?), otherwise new request
            response = requests.post(self.completion_url, json=payload, headers=self.headers, timeout=1200)
            
            if response.status_code != 200:
                return GenerationResult(text="", success=False, error_message=f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            output_text = data['choices'][0]['text']
            return GenerationResult(text=output_text, success=True)

        except Exception as e:
            logger.error(f"Error calling vLLM Sync: {e}")
            return GenerationResult(text="", success=False, error_message=str(e))
