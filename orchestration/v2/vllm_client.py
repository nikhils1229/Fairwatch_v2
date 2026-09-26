"""
FairWatch V2 - vLLM Client (orchestration/v2/vllm_client.py)
Unified, high-throughput client supporting OpenAI chat completions format,
structured JSON schema guided decoding, seeds, and strict error handling.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import requests
from openai import OpenAI

LOG = logging.getLogger(__name__)

# Strict JSON Schema for Domain Agent Loan Evaluations
LOAN_EVALUATION_SCHEMA = {
    "name": "loan_evaluation",
    "schema": {
        "type": "object",
        "properties": {
            "agent_name": {"type": "string"},
            "loan_type": {"type": "string"},
            "approval_decision": {"type": "string", "enum": ["approve", "deny"]},
            "approval_type": {"type": "string", "enum": ["STANDARD_TERMS", "CONDITIONAL_APPROVAL", "MANUAL_REVIEW", "DENIAL"]},
            "interest_rate": {"type": "number"},
            "confidence_probability": {"type": "integer"},
            "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "approval_decision_reason": {"type": "string"},
            "approval_type_reason": {"type": "string"},
            "interest_rate_reason": {"type": "string"},
            "confidence_reason": {"type": "string"}
        },
        "required": [
            "agent_name", "loan_type", "approval_decision", "approval_type",
            "interest_rate", "confidence_probability", "confidence_level",
            "approval_decision_reason", "approval_type_reason", "interest_rate_reason", "confidence_reason"
        ],
        "additionalProperties": False
    }
}

# Strict JSON Schema for Business Decision Synthesis / Judge
BUSINESS_SYNTHESIS_SCHEMA = {
    "name": "business_synthesis",
    "schema": {
        "type": "object",
        "properties": {
            "agent_name": {"type": "string"},
            "loan_type": {"type": "string"},
            "approval_decision": {"type": "string", "enum": ["approve", "deny"]},
            "approval_type": {"type": "string", "enum": ["STANDARD_TERMS", "CONDITIONAL_APPROVAL", "MANUAL_REVIEW", "DENIAL"]},
            "interest_rate": {"type": "number"},
            "confidence_probability": {"type": "integer"},
            "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "reasoning": {
                "type": "object",
                "properties": {
                    "approval_decision_reason": {"type": "string"},
                    "approval_type_reason": {"type": "string"},
                    "interest_rate_reason": {"type": "string"},
                    "confidence_reason": {"type": "string"}
                },
                "required": ["approval_decision_reason", "approval_type_reason", "interest_rate_reason", "confidence_reason"],
                "additionalProperties": False
            },
            "agent_influence": {
                "type": "object",
                "properties": {
                    "risk_manager_weight": {"type": "number"},
                    "regulatory_weight": {"type": "number"},
                    "data_science_weight": {"type": "number"},
                    "consumer_advocate_weight": {"type": "number"},
                    "primary_influence": {"type": "string", "enum": ["Risk Manager", "Regulatory Compliance", "Data Science", "Consumer Advocate"]}
                },
                "required": ["risk_manager_weight", "regulatory_weight", "data_science_weight", "consumer_advocate_weight", "primary_influence"],
                "additionalProperties": False
            }
        },
        "required": [
            "agent_name", "loan_type", "approval_decision", "approval_type",
            "interest_rate", "confidence_probability", "confidence_level",
            "reasoning", "agent_influence"
        ],
        "additionalProperties": False
    }
}

@dataclass
class GenerationResult:
    text: str
    success: bool
    error_message: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

class VLLMClientV2:
    """
    Synchronous vLLM client using OpenAI compatible API.
    Supports guided decoding, temperature, seed, and retry on transient errors.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8002/v1", model_alias: str = "llama8b", api_key: str = "EMPTY"):
        self.base_url = base_url.rstrip("/")
        self.model_alias = model_alias
        self.client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=120.0)
        self.current_seed: Optional[int] = None
        self.guided: bool = True

    def set_seed(self, seed: Optional[int]):
        self.current_seed = seed

    def set_guided(self, guided: bool):
        self.guided = guided

    def generate(self, prompt: str, system_prompt: Optional[str] = None, config: Any = None) -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temperature = getattr(config, "temperature", 0.0) if config else 0.0
        max_tokens = getattr(config, "max_tokens", 1024) if config else 1024
        seed = getattr(config, "seed", self.current_seed) if config else self.current_seed

        # Choose schema based on prompt intent
        is_synth = ("synthesize" in prompt.lower() or "business decision" in prompt.lower() or "agent recommendations" in prompt.lower())
        active_schema = BUSINESS_SYNTHESIS_SCHEMA if is_synth else LOAN_EVALUATION_SCHEMA

        response_format = None
        if self.guided:
            response_format = {"type": "json_schema", "json_schema": active_schema}

        max_attempts = 3
        last_exc = None
        for attempt in range(max_attempts):
            try:
                kwargs = {
                    "model": self.model_alias,
                    "messages": messages,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                }
                if seed is not None:
                    kwargs["seed"] = int(seed)
                if response_format is not None:
                    kwargs["response_format"] = response_format

                resp = self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                text = choice.message.content or ""
                prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) if hasattr(resp, "usage") else 0
                completion_tokens = getattr(resp.usage, "completion_tokens", 0) if hasattr(resp, "usage") else 0

                return GenerationResult(
                    text=text,
                    success=True,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
            except Exception as e:
                last_exc = e
                LOG.warning(f"vLLM call failed (attempt {attempt+1}/{max_attempts}): {e}")

        return GenerationResult(
            text="",
            success=False,
            error_message=str(last_exc)
        )
