import json
import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Union
import aiohttp
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    text: str
    success: bool
    error_message: Optional[str] = None

class AsyncVLLMClient:
    """
    Asynchronous vLLM (FastChat/OpenAI) Client using aiohttp.
    Designed for use with asyncio.Semaphore for concurrency.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY", model_name: str = "casperhansen/llama-3.2-3b-instruct-awq", session: Optional[aiohttp.ClientSession] = None):
        if session is None:
            raise ValueError("An aiohttp.ClientSession must be provided")
            
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        # Switch to OpenAI compatible chat completions endpoint if needed, but Completions is used by standard
        # Switch to OpenAI compatible chat completions endpoint
        self.completion_url = f"{base_url}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.session = session

    async def generate(self, prompt: str, system_prompt: str = None, config: Any = None, json_schema: Dict[str, Any] = None) -> GenerationResult:
        """
        Async generation leveraging Chat completions API for accurate tokenizer mapping.
        """
        try:
            # 1. Message Construction
            # Gemma 2 natively struggles with 'system' roles in some inference engines.
            # Best practice is to prepend system constraints directly to the user prompt.
            if system_prompt:
                combined_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                combined_prompt = prompt
                
            messages = [{"role": "user", "content": combined_prompt}]

            # 2. Config Extraction
            temperature = config.temperature if config and hasattr(config, 'temperature') else 0.7
            max_tokens = config.max_tokens if config and hasattr(config, 'max_tokens') else 400
            
            # 3. Payload Construction
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # 4. Guided Decoding (Structured JSON Enforcement via Response Format)
            if json_schema:
                formal_schema = self._generate_json_schema(json_schema)
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema",
                        "schema": formal_schema
                    }
                }

            # 5. Async Request — with retry + 120s timeout
            _MAX_RETRIES = 3
            _BASE_DELAY  = 2.0  # seconds; doubles each attempt
            _timeout     = aiohttp.ClientTimeout(total=120)

            last_error = ""
            for attempt in range(_MAX_RETRIES):
                try:
                    async with self.session.post(
                        self.completion_url,
                        json=payload,
                        headers=self.headers,
                        timeout=_timeout
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            last_error = f"HTTP {response.status}: {error_text}"
                            # Don't retry on hard 4xx errors
                            if response.status < 500:
                                return GenerationResult(text="", success=False, error_message=last_error)
                            # 5xx — retry
                        else:
                            data = await response.json()
                            output_text = data['choices'][0]['message']['content']
                            return GenerationResult(text=output_text, success=True)

                except asyncio.TimeoutError:
                    last_error = f"Timeout on attempt {attempt+1} (60s limit)"
                    logger.warning(f"{last_error} → {self.completion_url}")
                except aiohttp.ClientConnectorError as e:
                    last_error = f"Connection error on attempt {attempt+1}: {e}"
                    logger.warning(last_error)

                if attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.0f}s (attempt {attempt+2}/{_MAX_RETRIES})…")
                    await asyncio.sleep(delay)

            return GenerationResult(text="", success=False, error_message=f"Failed after {_MAX_RETRIES} attempts: {last_error}")

        except Exception as e:
            logger.error(f"Error calling vLLM Async: {str(e)}")
            return GenerationResult(text="", success=False, error_message=str(e))

    def _generate_json_schema(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively converts a prompt-template dictionary into a formal JSON Schema Draft 7.
        Handles nested objects, basic types, and placeholder strings like "<float: 2.0-30.0>".
        """
        if not isinstance(template, dict):
            return {"type": "string"} # Fallback

        schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }

        for key, value in template.items():
            schema["required"].append(key)
            
            if isinstance(value, dict):
                # Recursively handle nested objects
                schema["properties"][key] = self._generate_json_schema(value)
            elif isinstance(value, str):
                v_lower = value.lower()
                if "<float" in v_lower or "number" in v_lower:
                    schema["properties"][key] = {"type": "number"}
                elif "<int" in v_lower or "integer" in v_lower:
                    schema["properties"][key] = {"type": "integer"}
                elif v_lower == "string":
                    schema["properties"][key] = {"type": "string"}
                else:
                    # Treat as string by default
                    schema["properties"][key] = {"type": "string"}
            else:
                # Direct type mapping
                if isinstance(value, float):
                    schema["properties"][key] = {"type": "number"}
                elif isinstance(value, int) and not isinstance(value, bool):
                    schema["properties"][key] = {"type": "integer"}
                elif isinstance(value, bool):
                    schema["properties"][key] = {"type": "boolean"}
                else:
                    schema["properties"][key] = {"type": "string"}

        return schema
