"""
Base Agent Class
Loads templates and personas from external files
"""

import json
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from orchestration.clients.ollama_client import OllamaClient, GenerationConfig

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

def _reject_constant(c):
    raise ValueError(f"Constant '{c}' not allowed in JSON")

def _sanitize_no_decimals(obj):
    if isinstance(obj, Decimal):
        if obj.is_nan() or obj.is_infinite() or obj.adjusted() > 308:
            return None
        try:
            if obj == obj.to_integral_value():
                return int(obj)
            f = float(obj)
            return f if math.isfinite(f) else None
        except (OverflowError, ValueError):
            return None
    elif isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    elif isinstance(obj, dict):
        return {k: _sanitize_no_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_no_decimals(v) for v in obj]
    return obj


# parse_status values written onto every record.
#   ok           - an approval_decision was read from the generation
#   unparseable  - the generation produced no readable decision
#   error        - generation itself failed (no usable text after retries)
# Records that are not 'ok' carry approval_decision=None and must be excluded
# from approval-rate denominators, not counted as denials.
from orchestration.harness.parse_status import PARSE_OK, PARSE_UNPARSEABLE, PARSE_ERROR, PARSE_REPAIRED_NULL, PARSE_REPAIRED_FUZZY, PARSE_FALLBACK


def normalize_decision(raw: Any) -> Optional[str]:
    """
    Map a raw approval_decision onto 'approve' / 'deny', or None if the
    model did not express an exact, unambiguous decision.
    """
    if raw is None:
        return None
    token = str(raw).strip().strip('.,;:!"\'').lower().replace('_', ' ')
    if token in ('approve', 'approved', 'approval', 'i approve', 'recommend approval', 'decision: approve'):
        return 'approve'
    if token in ('deny', 'denied', 'denial', 'decline', 'declined', 'reject', 'rejected',
                 'i deny', 'recommend denial', 'decision: deny'):
        return 'deny'

    # Any other prose or ambiguous string returns None (model failed exact output contract)
    return None


class BaseAgent(ABC):
    """Base class for all credit evaluation agents"""
    
    def __init__(self, client: OllamaClient, agent_name: str, template_path: str = None):
        """
        Initialize base agent
        
        Args:
            client: OllamaClient instance
            agent_name: Name of agent (used to load persona)
        """
        self.client = client
        self.agent_name = agent_name
        
        # Determine paths relative to project root
        project_root = Path(__file__).parent.parent
        if template_path is None:
            template_path = project_root / "templates" / "loan_evaluation_template.json"
        
        persona_path = project_root / "prompts" / f"{agent_name}_persona.txt"
        
        # Load template and persona
        self.template = self._load_template(str(template_path))
        self.persona = self._load_persona(str(persona_path))
        
        # Build system prompt
        self.system_prompt = self._build_system_prompt()
        
        # Default generation config
        self.config = GenerationConfig(temperature=0.7, max_tokens=500)
    
    def _load_template(self, template_path: str) -> Dict[str, Any]:
        """Load JSON template from file"""
        try:
            path = Path(template_path)
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            LOG.error(f"Template not found: {template_path}")
            raise
        except json.JSONDecodeError:
            LOG.error(f"Invalid JSON in template: {template_path}")
            raise
    
    def _load_persona(self, persona_path: str) -> str:
        """Load persona description from file"""
        try:
            path = Path(persona_path)
            with open(path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            LOG.error(f"Persona not found: {persona_path}")
            raise
    
    def _build_system_prompt(self) -> str:
        """Build system prompt from persona and template"""
        schema = self.template['output_schema']
        definitions = self.template.get('approval_type_definitions', {})
        confidence_guidance = self.template.get('confidence_guidance', {})
        
        # Build approval type definitions section
        approval_types_text = "\n".join([
            f"- {k}: {v}" for k, v in definitions.items()
        ])
        
        # Build confidence guidance section
        confidence_text = f"""
Confidence Scoring Guidance:
- {confidence_guidance.get('instruction', 'Use full 0-100 range')}
- High (80-100%): {confidence_guidance.get('high_confidence', 'Clear decision')}
- Medium (40-60%): {confidence_guidance.get('medium_confidence', 'Uncertain')}
- Low (0-39%): {confidence_guidance.get('low_confidence', 'Significant concerns')}
"""
        
        system_prompt = f"""{self.persona}

Respond ONLY with valid JSON. All fields are required. No markdown, no extra text.

RULES: approval_decision={{'"approve"'|'"deny"'}}, approval_type={{'"STANDARD_TERMS"'|'"CONDITIONAL_APPROVAL"'|'"DENIAL"'}}, interest_rate=float {self.template['validation_rules']['interest_rate_range'][0]}-{self.template['validation_rules']['interest_rate_range'][1]}, confidence_probability=int 0-100, confidence_level={{'"low"'|'"medium"'|'"high"'}}. Every field is required.

{confidence_text}
OUTPUT SCHEMA -- replace every <...> placeholder with your own value.
Do not echo the placeholders and do not treat them as suggested answers:
{{
  "agent_name": "<your agent name>",
  "loan_type": "<loan type>",
  "approval_decision": "<approve|deny>",
  "approval_type": "<STANDARD_TERMS|CONDITIONAL_APPROVAL|DENIAL>",
  "interest_rate": <float>,
  "confidence_probability": <int 0-100>,
  "confidence_level": "<low|medium|high>",
  "approval_decision_reason": "<your own reasoning>",
  "approval_type_reason": "<your own reasoning>",
  "interest_rate_reason": "<your own reasoning>",
  "confidence_reason": "<your own reasoning>"
}}
"""
        return system_prompt
    
    def _clean_json(self, text: str) -> str:
        """Clean JSON response from potential markdown formatting"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```', '', text)
        
        # Extract JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1]
        
        return text
    
    def _normalize_decision(self, raw: Any) -> str:
        """Instance-level alias for :func:`normalize_decision`."""
        return normalize_decision(raw)

    def _fix_null_values(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a parsed generation into the record schema.

        Coerces only what can be read. Anything unreadable becomes None and is
        flagged via `parse_status`; nothing is imputed. A record leaving this
        method carries `parse_status` of 'ok' or 'unparseable', so downstream
        validity is computable from the record alone rather than by matching
        sentinel strings.
        """
        # Top-level guard: a non-dict generation carries no decision at all.
        if not isinstance(parsed, dict):
            return {
                "agent_name": self.agent_name.replace('_', ' ').title(),
                "loan_type": "personal_loan",
                "approval_decision": None,
                "approval_type": None,
                "interest_rate": None,
                "confidence_probability": None,
                "confidence_level": None,
                "parse_status": PARSE_UNPARSEABLE,
                "parse_detail": "non-dict generation",
                "raw_excerpt": str(parsed)[:500],
                "reasoning": {
                    "approval_decision_reason": "Not provided",
                    "approval_type_reason": "Not provided",
                    "interest_rate_reason": "Not provided",
                    "confidence_reason": "Not provided",
                },
            }

        unreadable = []

        # --- approval_decision ---
        raw_decision = parsed.get('approval_decision')
        decision = self._normalize_decision(raw_decision)
        parsed['approval_decision'] = decision
        if decision is None:
            unreadable.append('approval_decision')

        # --- approval_type ---
        # Never derived from a decision we could not read. Deriving 'DENIAL'
        # from a null decision is how a parse failure became a loan denial.
        raw_type = str(parsed.get('approval_type', '')).strip().strip('.').upper()
        if decision == 'approve':
            if raw_type in {'STANDARD_TERMS', 'CONDITIONAL_APPROVAL', 'MANUAL_REVIEW'}:
                parsed['approval_type'] = raw_type
            elif raw_type in {'STANDARD', 'STANDARD TERMS'}:
                parsed['approval_type'] = 'STANDARD_TERMS'
            elif raw_type in {'CONDITIONAL', 'CONDITIONAL APPROVAL'}:
                parsed['approval_type'] = 'CONDITIONAL_APPROVAL'
            elif raw_type in {'MANUAL', 'MANUAL REVIEW'}:
                parsed['approval_type'] = 'MANUAL_REVIEW'
            else:
                parsed['approval_type'] = None
                unreadable.append('approval_type')
        elif decision == 'deny':
            if raw_type in {'DENIAL', 'DENY'}:
                parsed['approval_type'] = 'DENIAL'
            else:
                parsed['approval_type'] = None
                unreadable.append('approval_type')
        else:
            parsed['approval_type'] = None
            if raw_type:
                unreadable.append('approval_type')

        # --- confidence_probability ---
        # None, not 0. Zero is a legal confidence value; imputing it moves the
        # mean of every confidence distribution toward zero by the failure rate.
        raw_conf = parsed.get('confidence_probability')
        confidence = None
        if raw_conf is not None:
            try:
                if isinstance(raw_conf, bool):
                    raise TypeError('bool is not a confidence')
                if isinstance(raw_conf, (int, float, Decimal)):
                    d_val = Decimal(str(raw_conf)) if not isinstance(raw_conf, Decimal) else raw_conf
                    if d_val.is_signed() and d_val == 0:
                        raise ValueError('negative zero confidence')
                    if Decimal('0.0') < d_val < Decimal('1.0'):
                        t = d_val.as_tuple()
                        d_val = Decimal((t.sign, t.digits, t.exponent + 2))
                    if not (Decimal('0.0') <= d_val <= Decimal('100.0')):
                        raise ValueError('out of exact bounds')
                    confidence = int(d_val.to_integral_value(rounding=ROUND_HALF_EVEN))
                else:
                    match = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\s*(%)?\s*$', str(raw_conf))
                    if not match:
                        raise ValueError('malformed confidence string')
                    num_str, pct_flag = match.group(1), match.group(2)
                    d_val = Decimal(num_str)
                    if d_val.is_signed() and d_val == 0:
                        raise ValueError('negative zero confidence')
                    if Decimal('0.0') < d_val < Decimal('1.0') and not pct_flag:
                        t = d_val.as_tuple()
                        d_val = Decimal((t.sign, t.digits, t.exponent + 2))
                    if not (Decimal('0.0') <= d_val <= Decimal('100.0')):
                        raise ValueError('out of exact bounds')
                    confidence = int(d_val.to_integral_value(rounding=ROUND_HALF_EVEN))
            except (TypeError, ValueError, OverflowError, InvalidOperation):
                confidence = None
        parsed['confidence_probability'] = confidence
        if confidence is None:
            unreadable.append('confidence_probability')
            LOG.warning(f"{self.agent_name}: unreadable confidence_probability -> None")

        # --- confidence_level ---
        raw_level = str(parsed.get('confidence_level', '')).strip().strip('.').lower()
        if raw_level in ('low', 'medium', 'high'):
            parsed['confidence_level'] = raw_level
        elif confidence is not None:
            parsed['confidence_level'] = (
                'low' if confidence < 40 else 'medium' if confidence < 80 else 'high'
            )
        else:
            parsed['confidence_level'] = None
            unreadable.append('confidence_level')

        # --- interest_rate ---
        # The old fallback was `25.0 if deny else 10.0`, which manufactured the
        # rate the prompt had just suggested for denials. Unreadable is None.
        raw_rate = parsed.get('interest_rate')
        rate = None
        rate_min, rate_max = 0.0, 40.0
        if isinstance(getattr(self, 'template', None), dict):
            rules = self.template.get('validation_rules', {})
            r_range = rules.get('interest_rate_range')
            if isinstance(r_range, (list, tuple)) and len(r_range) == 2:
                try:
                    rate_min, rate_max = float(r_range[0]), float(r_range[1])
                except (ValueError, TypeError):
                    pass
        if raw_rate is not None:
            try:
                if isinstance(raw_rate, bool):
                    raise TypeError('bool is not a rate')
                if isinstance(raw_rate, (int, float, Decimal)):
                    f = float(raw_rate)
                    if math.isfinite(f) and rate_min <= f <= rate_max:
                        rate = f
                else:
                    cleaned_s = str(raw_rate).strip().replace('%', '')
                    match = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\s*$', cleaned_s)
                    if match:
                        f = float(match.group(1))
                        if math.isfinite(f) and rate_min <= f <= rate_max:
                            rate = f
            except (TypeError, ValueError, OverflowError):
                rate = None
        parsed['interest_rate'] = rate
        if rate is None:
            unreadable.append('interest_rate')

        # --- reasoning ---
        # Generations are flat; the stored record nests. Accept either shape.
        reasoning_sub = parsed.get('reasoning', {})
        if not isinstance(reasoning_sub, dict):
            reasoning_sub = {}

        final_reasoning = {
            k: parsed.get(k, reasoning_sub.get(k, 'Not provided'))
            for k in ('approval_decision_reason', 'approval_type_reason',
                      'interest_rate_reason', 'confidence_reason')
        }
        for k, v in final_reasoning.items():
            if v is None or not str(v).strip():
                final_reasoning[k] = "Not provided"

        parsed['reasoning'] = final_reasoning
        for k in final_reasoning:
            parsed.pop(k, None)

        # --- parse_status ---
        # Written on every record, successes included, so future runs are
        # auditable without sentinel-string matching.
        if decision is None:
            parsed['parse_status'] = PARSE_UNPARSEABLE
            parsed['parse_detail'] = "unreadable: " + ", ".join(unreadable)
            if raw_decision is not None:
                parsed['raw_excerpt'] = str(raw_decision)[:500]
            LOG.warning(
                f"{self.agent_name}: no readable approval_decision "
                f"(raw={str(raw_decision)[:80]!r}) -> None, not 'deny'"
            )
        else:
            parsed['parse_status'] = PARSE_OK
            if unreadable:
                parsed['parse_detail'] = "unreadable: " + ", ".join(unreadable)

        return _sanitize_no_decimals(parsed)

    def _safe_float(self, v: Any, default: float = 0.0) -> float:
        """Safely convert a value to float, handling percentages and None."""
        try:
            if v is None: return default
            if isinstance(v, (int, float)): return float(v)
            return float(str(v).strip().replace('%', ''))
        except (ValueError, TypeError, OverflowError):
            return default

    def _safe_int(self, v: Any, default: int = 0) -> int:
        """Safely convert a value to int, handling percentages and None."""
        try:
            if v is None: return default
            if isinstance(v, (int, float)): return int(v)
            return int(float(str(v).strip().replace('%', '')))
        except (ValueError, TypeError, OverflowError):
            return default
    
    def evaluate_loan_application(self, application_data: str) -> Dict[str, Any]:
        """
        Evaluate a loan application
        
        Args:
            application_data: Loan application prompt text
            
        Returns:
            Dictionary with evaluation results
        """
        first_attempt_text = ""
        last_text = ""
        last_error_msg = None

        # Attempt 1: Initial generation
        res = self.client.generate(
            prompt=application_data,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        if res.success:
            try:
                cleaned = self._clean_json(res.text)
                parsed = json.loads(cleaned, parse_float=Decimal, parse_int=Decimal, parse_constant=_reject_constant)
                
                # Fix null values
                parsed = self._fix_null_values(parsed)
                
                # Add agent name and loan type
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                parsed['attempt'] = 1
                parsed['retry_reason'] = None
                parsed['raw_response'] = res.text
                
                return parsed
            except (json.JSONDecodeError, ValueError) as e:
                LOG.warning(f"{self.agent_name} JSON parse failed (attempt 1): {e}")
                first_attempt_text = res.text
        else:
            last_error_msg = getattr(res, "error_message", None)
        
        # Attempt 2: Retry with error feedback
        retry_prompt = f"""You are acting as the {self.agent_name.replace('_', ' ').title()}. Your previous response was invalid JSON. 
        
APPLICATION DATA:
{application_data}

Maintain your strict persona criteria. Respond with ONLY valid JSON matching the required schema.
CRITICAL: every field must carry your own value. Do not use null.

No explanation, just JSON."""
        
        res = self.client.generate(
            prompt=retry_prompt,
            system_prompt=self.system_prompt,
            config=self.config
        )
        
        if res.success:
            try:
                cleaned = self._clean_json(res.text)
                parsed = json.loads(cleaned, parse_float=Decimal, parse_int=Decimal, parse_constant=_reject_constant)
                
                # Fix null values
                parsed = self._fix_null_values(parsed)
                
                # Add agent name and loan type
                parsed['agent_name'] = self.agent_name.replace('_', ' ').title()
                parsed['loan_type'] = 'personal_loan'
                parsed['attempt'] = 2
                parsed['retry_reason'] = 'json_syntax_error'
                parsed['raw_response'] = res.text
                
                return parsed
            except (json.JSONDecodeError, ValueError) as e:
                LOG.error(f"{self.agent_name} JSON parse failed (attempt 2): {e}")
                LOG.error(f"Raw response: {res.text}")
                last_text = res.text
        else:
            last_error_msg = getattr(res, "error_message", None) or last_error_msg
        
        # Both attempts failed to yield JSON or generation failed.
        # Preserve client error message if present so wrapper can classify error taxonomy.
        err_msg = last_error_msg or "Failed to generate valid JSON after 2 attempts"
        return self._create_error_response(
            err_msg,
            raw_excerpt=last_text or first_attempt_text,
        )
    
    def _create_error_response(self, error_message: str,
                               raw_excerpt: str = "") -> Dict[str, Any]:
        """
        Record for a generation that never produced usable text.

        This used to return approval_decision='deny' with a "System Error:"
        string buried in the reasoning. 103 stored result files contain that
        sentinel, and every one of them counts as a denial in any analysis that
        reads approval_decision. An infrastructure failure is not a credit
        decision, so the decision is None and parse_status says why.
        """
        record = {
            "agent_name": self.agent_name.replace('_', ' ').title(),
            "loan_type": "personal_loan",
            "approval_decision": None,
            "approval_type": None,
            "interest_rate": None,
            "confidence_probability": None,
            "confidence_level": None,
            "parse_status": PARSE_ERROR,
            "parse_detail": error_message,
            "attempt": 2,
            "reasoning": {
                "approval_decision_reason": "Not provided",
                "approval_type_reason": "Not provided",
                "interest_rate_reason": "Not provided",
                "confidence_reason": "Not provided"
            }
        }
        if raw_excerpt:
            record["raw_excerpt"] = raw_excerpt[:500]
            record["raw_response"] = raw_excerpt
        record["retry_reason"] = "json_syntax_error"
        return record
    
    def process(self, text: str) -> str:
        """
        Process method for ConversationChain compatibility
        Returns JSON string
        """
        result = self.evaluate_loan_application(text)
        return json.dumps(result, indent=2)

    @abstractmethod
    def evaluate_pairwise_comparison(self, context: str, current_prompt: str) -> str:
        """
        To be implemented by subclasses if pairwise evaluation is needed.
        """
        pass