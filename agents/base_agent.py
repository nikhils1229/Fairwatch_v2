import json
import logging
import re
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union

try:
    import json_repair
except ImportError:
    json_repair = None

from utils.async_client import AsyncLLMClient, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all credit evaluation agents.
    Refactored to be fully asynchronous and include character-preserving healing.
    """
    
    def __init__(
        self, 
        client: AsyncLLMClient, 
        agent_name: str, 
        template_path: Optional[str] = None,
        persona_path: Optional[str] = None
    ):
        self.client = client
        self.agent_name = agent_name
        
        project_root = Path(__file__).parent.parent
        
        if template_path is None:
            template_path = project_root / "templates" / "loan_evaluation_template.json"
        if persona_path is None:
            persona_path = project_root / "prompts" / f"{agent_name}_persona.txt"
            
        self.template = self._load_json(str(template_path))
        self.persona = self._load_text(str(persona_path))
        
        self.system_prompt = self._build_system_prompt()
        # max_tokens=256 is enough for the agent JSON outputs (typical
        # ~150-200 tokens). Cap reduces wallclock ~2x with no observed
        # truncation; json_repair handles partial JSON if it ever clips.
        self.config = GenerationConfig(temperature=0.7, max_tokens=256)

    def _load_json(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {path}: {e}")
            raise

    def _load_text(self, path: str) -> str:
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load text from {path}: {e}")
            raise

    def _build_system_prompt(self) -> str:
        schema = self.template.get('output_schema', {})
        definitions = self.template.get('approval_type_definitions', {})
        confidence_guidance = self.template.get('confidence_guidance', {})
        
        approval_types_text = "\n".join([f"- {k}: {v}" for k, v in definitions.items()])
        
        confidence_text = f"""
Confidence Scoring Guidance:
- {confidence_guidance.get('instruction', 'Use full 0-100 range')}
- High (80-100%): {confidence_guidance.get('high_confidence', 'Clear decision')}
- Medium (40-60%): {confidence_guidance.get('medium_confidence', 'Uncertain')}
- Low (0-39%): {confidence_guidance.get('low_confidence', 'Significant concerns')}
"""
        
        system_prompt = f"""{self.persona}

You MUST respond in valid JSON matching this exact schema:
{json.dumps(schema, indent=2)}

Approval Type Definitions:
{approval_types_text}

{confidence_text}

CRITICAL RULES:
- Output ONLY valid JSON.
- Include ALL fields from schema.
- approval_decision must be "approve" or "deny".
- interest_rate must be a float between {self.template['validation_rules']['interest_rate_range'][0]} and {self.template['validation_rules']['interest_rate_range'][1]}.
- confidence_probability must be an integer 0-100.
- reasoning must have all 4 sub-fields.
- NEVER use null or omit fields.
"""
        return system_prompt

    def _heal_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Character-preserving healing logic.
        Uses json_repair to fix syntax and ensures semantic integrity.
        """
        # 1. Basic cleaning
        text = re.sub(r'```json\s*', '', raw_text)
        text = re.sub(r'```', '', text)
        
        # 2. Syntax repair
        if json_repair:
            repaired_json = json_repair.repair_json(text)
        else:
            # Fallback: try to extract JSON object
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                repaired_json = text[start:end+1]
            else:
                repaired_json = text

        try:
            parsed = json.loads(repaired_json)
        except Exception as e:
            # If standard json fails, try a very basic fix for common issues if json_repair is missing
            if not json_repair:
                try:
                    # B2 fix: only quote keys that are NOT already quoted
                    # Negative-lookbehind prevents double-quoting "key":
                    fixed = re.sub(r'(?<!")\b(\w+)(?!")\s*:', r'"\1":', repaired_json)
                    # Fix trailing commas before closing braces/brackets
                    fixed = re.sub(r',\s*}', '}', fixed)
                    fixed = re.sub(r',\s*]', ']', fixed)
                    parsed = json.loads(fixed)
                except:
                    logger.warning(f"JSON parsing failed: {e}")
                    return self._create_error_response("Invalid JSON structure")
            else:
                logger.warning(f"JSON parsing failed: {e}")
                return self._create_error_response("Invalid JSON structure")

        if not isinstance(parsed, dict):
            return self._create_error_response("Response is not a JSON object")

        # 3. Semantic preservation and type enforcement
        # We try to preserve the agent's intent while ensuring schema compliance.
        
        # Decision normalization
        decision = str(parsed.get('approval_decision', 'deny')).lower()
        if 'approve' in decision:
            parsed['approval_decision'] = 'approve'
        else:
            parsed['approval_decision'] = 'deny'
            
        # Type normalization
        app_type = str(parsed.get('approval_type', 'DENIAL')).upper()
        if parsed['approval_decision'] == 'approve':
            if 'CONDITIONAL' in app_type:
                parsed['approval_type'] = 'CONDITIONAL_APPROVAL'
            elif 'SUBOPTIMAL' in app_type:
                parsed['approval_type'] = 'SUBOPTIMAL_TERMS'
            else:
                parsed['approval_type'] = 'STANDARD_TERMS'
        else:
            parsed['approval_type'] = 'DENIAL'

        # Interest rate enforcement
        rate = parsed.get('interest_rate')
        parsed['interest_rate'] = self._to_float(rate, default=25.0 if parsed['approval_decision'] == 'deny' else 10.0)
        
        # Confidence enforcement
        prob = parsed.get('confidence_probability')
        parsed['confidence_probability'] = int(self._to_float(prob, default=50.0))
        parsed['confidence_probability'] = max(0, min(100, parsed['confidence_probability']))
        
        if parsed.get('confidence_level') not in ['low', 'medium', 'high']:
            if parsed['confidence_probability'] < 40:
                parsed['confidence_level'] = 'low'
            elif parsed['confidence_probability'] < 80:
                parsed['confidence_level'] = 'medium'
            else:
                parsed['confidence_level'] = 'high'

        # B3 fix: track which fields were actually promoted from the flat top-level
        # so we only delete those, not fields that were already in a valid reasoning dict.
        reasoning_was_valid_dict = isinstance(parsed.get('reasoning'), dict)
        
        reasoning = parsed.get('reasoning', {})
        if not isinstance(reasoning, dict):
            # Maybe it's flat?
            reasoning = {
                "approval_decision_reason": parsed.get("approval_decision_reason", "Not provided"),
                "approval_type_reason": parsed.get("approval_type_reason", "Not provided"),
                "interest_rate_reason": parsed.get("interest_rate_reason", "Not provided"),
                "confidence_reason": parsed.get("confidence_reason", "Not provided")
            }
        
        required_reasoning = [
            "approval_decision_reason", 
            "approval_type_reason", 
            "interest_rate_reason", 
            "confidence_reason"
        ]
        promoted_fields = []  # fields lifted from top-level into reasoning
        for field in required_reasoning:
            if field not in reasoning or not reasoning[field]:
                # Try to find it in the top level if it was flat
                top_val = parsed.get(field)
                if top_val:
                    reasoning[field] = top_val
                    promoted_fields.append(field)
                else:
                    reasoning[field] = "Not provided"
        
        parsed['reasoning'] = reasoning
        
        # Only remove top-level keys that were actually promoted into the reasoning block.
        # If reasoning was already a valid dict, don't touch the top-level keys.
        if not reasoning_was_valid_dict:
            for field in promoted_fields:
                if field in parsed:
                    del parsed[field]

        # Final metadata
        parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
        parsed['loan_type'] = 'personal_loan'
        
        return parsed

    def _to_float(self, val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        try:
            # Handle strings like "8.5%" or "10.0 USD"
            match = re.search(r'(\d+(\.\d+)?)', str(val))
            if match:
                return float(match.group(1))
        except:
            pass
        return default

    async def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Evaluate a loan application asynchronously.
        """
        res = await self.client.generate(
            prompt=application_data,
            system_prompt=self.system_prompt,
            config=self.config,
            json_schema=self.template.get('output_schema')
        )
        
        if res.success:
            return self._heal_json(res.text)
        
        # Retry once if failed
        logger.warning(f"Initial generation failed for {self.agent_name}: {res.error_message}. Retrying...")
        
        res = await self.client.generate(
            prompt=f"Your previous response was invalid. Please evaluate this application again and return ONLY valid JSON.\n\n{application_data}",
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        if res.success:
            return self._heal_json(res.text)
            
        return self._create_error_response(f"LLM generation failed: {res.error_message}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name.replace('_', ' ').title(),
            "loan_type": "personal_loan",
            "approval_decision": "deny",
            "approval_type": "DENIAL",
            "interest_rate": 25.0,
            "confidence_probability": 0,
            "confidence_level": "low",
            "reasoning": {
                "approval_decision_reason": f"System Error: {error_message}",
                "approval_type_reason": "N/A",
                "interest_rate_reason": "N/A",
                "confidence_reason": "N/A"
            }
        }

    @abstractmethod
    async def evaluate_pairwise_comparison(self, context: str, current_prompt: str) -> Dict[str, Any]:
        """
        To be implemented by subclasses.
        """
        pass
