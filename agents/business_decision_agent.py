"""
Business Decision Agent - Robust JSON parsing and valid-only metric compliance
"""

import json
import logging
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from agents.base_agent import normalize_decision, PARSE_OK, PARSE_UNPARSEABLE, PARSE_ERROR, _reject_constant, _sanitize_no_decimals

LOG = logging.getLogger(__name__)

# Helpers _reject_constant and _sanitize_no_decimals imported from base_agent



class BusinessDecisionAgent:
    """Synthesizes final decision from multiple agent recommendations"""

    def __init__(self, client):
        self.client = client
        self.agent_name = "business_decision"

        # Anchor paths relative to repo root
        project_root = Path(__file__).resolve().parent.parent
        persona_path = project_root / "prompts" / "business_decision_persona.txt"
        if persona_path.exists():
            with open(persona_path, "r", encoding="utf-8") as f:
                self.persona = f.read().strip()
        else:
            self.persona = "You are a business decision maker synthesizing loan recommendations."

        template_path = project_root / "templates" / "business_synthesis_template.json"
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                self.template = json.load(f)
        else:
            self.template = {}

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        """Safely convert to float, rejecting NaN/Inf/bool."""
        if value is None or isinstance(value, bool):
            return default
        try:
            val = float(str(value).strip().replace('%', ''))
            return val if (math.isfinite(val) and val >= 0) else default
        except (ValueError, TypeError, OverflowError):
            return default

    def _safe_int(self, value: Any, default: Optional[int] = None) -> Optional[int]:
        """Safely convert to int with Decimal exact domain and rounding."""
        if value is None or isinstance(value, bool):
            return default
        try:
            if isinstance(value, (int, float, Decimal)):
                d_val = Decimal(str(value)) if not isinstance(value, Decimal) else value
                if d_val.is_signed() and d_val == 0:
                    return default
                if Decimal('0.0') < d_val < Decimal('1.0'):
                    t = d_val.as_tuple()
                    d_val = Decimal((t.sign, t.digits, t.exponent + 2))
                if 0.0 <= d_val <= 100.0:
                    return int(d_val.to_integral_value(rounding=ROUND_HALF_EVEN))
                return default
            match = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\s*(%)?\s*$', str(value))
            if not match:
                return default
            num_str, pct_flag = match.group(1), match.group(2)
            d_val = Decimal(num_str)
            if d_val.is_signed() and d_val == 0:
                return default
            if Decimal('0.0') < d_val < Decimal('1.0') and not pct_flag:
                t = d_val.as_tuple()
                d_val = Decimal((t.sign, t.digits, t.exponent + 2))
            if not (Decimal('0.0') <= d_val <= Decimal('100.0')):
                return default
            return int(d_val.to_integral_value(rounding=ROUND_HALF_EVEN))
        except (ValueError, TypeError, OverflowError, InvalidOperation):
            return default

    def synthesize_decision(self, application_data: str, agent_recommendations: list) -> dict:
        """
        Synthesize final decision from agent recommendations
        """
        formatted_recs = []
        agent_names = ['Risk Manager', 'Regulatory Compliance', 'Data Science', 'Consumer Advocate']

        for i, (name, rec) in enumerate(zip(agent_names, agent_recommendations)):
            if not isinstance(rec, dict):
                rec = {}
            decision = rec.get('approval_decision', 'N/A')
            app_type = rec.get('approval_type', 'N/A')
            rate = self._safe_float(rec.get('interest_rate'))
            rate_str = f"{rate}%" if rate is not None else "N/A"
            conf_prob = self._safe_int(rec.get('confidence_probability'))
            conf_str = f"{conf_prob}%" if conf_prob is not None else "N/A"
            conf_level = rec.get('confidence_level', 'N/A')
            reasoning = rec.get('reasoning', {})
            reason_text = reasoning.get('approval_decision_reason', 'N/A') if isinstance(reasoning, dict) else 'N/A'

            rec_text = f"""
Agent {i+1}: {name}
- Decision: {decision}
- Type: {app_type}
- Interest Rate: {rate_str}
- Confidence: {conf_str} ({conf_level})
- Reasoning: {reason_text}
"""
            formatted_recs.append(rec_text.strip())

        recommendations_text = "\n\n".join(formatted_recs)

        system_instructions = "You are a business decision agent. Return ONLY valid JSON, no markdown, no other text."

        prompt = f"""{system_instructions}

{self.persona}

LOAN APPLICATION:
{application_data}

AGENT RECOMMENDATIONS:
{recommendations_text}

TASK:
Synthesize a final business decision considering all agent perspectives.

SYNTHESIS INSTRUCTIONS:
- Review all 4 agent recommendations and synthesize into final business decision
- Risk Manager: High weight on default probability and portfolio protection
- Regulatory: Veto power on compliance issues - must be satisfied
- Data Science: Important for predictive accuracy - but not sole factor
- Consumer Advocate: Balance against risk - important for reputation
- If all agents agree, follow consensus with high confidence
- If agents split, make judgment call and explain reasoning clearly

Return ONLY valid JSON matching this structure (no markdown, no backticks):
{{
  "agent_name": "Business Decision",
  "loan_type": "personal_loan",
  "approval_decision": "approve or deny",
  "approval_type": "STANDARD_TERMS or CONDITIONAL_APPROVAL or MANUAL_REVIEW or DENIAL",
  "interest_rate": 8.5,
  "confidence_probability": 75,
  "confidence_level": "high",
  "agent_influence": {{
    "risk_manager_weight": 0.25,
    "regulatory_weight": 0.25,
    "data_science_weight": 0.25,
    "consumer_advocate_weight": 0.25,
    "primary_influence": "Name of most influential agent"
  }},
  "reasoning": {{
    "synthesis_rationale": "Why this decision",
    "weight_justification": "Why these weights",
    "risk_assessment": "Final risk view"
  }}
}}

CRITICAL: Return ONLY the JSON object. Weights must sum to 1.0.
"""

        try:
            result = self.client.generate(prompt=prompt)
            response = result.text if hasattr(result, 'text') else str(result)
        except Exception as e:
            LOG.error(f"Generation error in BusinessDecisionAgent: {e}")
            return self._create_error_response(f"Generation error: {str(e)}")

        # Parse JSON
        try:
            business_decision = json.loads(response, parse_float=Decimal, parse_int=Decimal, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError, TypeError):
            response_clean = str(response).strip()
            if "```json" in response_clean:
                json_text = response_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in response_clean:
                json_text = response_clean.split("```")[1].split("```")[0].strip()
            else:
                start = response_clean.find('{')
                end = response_clean.rfind('}') + 1
                if start >= 0 and end > start:
                    json_text = response_clean[start:end]
                else:
                    return self._create_error_response(
                        "Could not extract JSON from response",
                        raw_excerpt=response_clean[:500]
                    )

            try:
                business_decision = json.loads(json_text, parse_float=Decimal, parse_int=Decimal, parse_constant=_reject_constant)
            except Exception as e:
                return self._create_error_response(
                    f"JSON parse failure: {e}",
                    raw_excerpt=response_clean[:500]
                )

        if not isinstance(business_decision, dict):
            return self._create_error_response(
                "Response is not a JSON object",
                raw_excerpt=str(response)
            )

        # Normalize decision
        raw_decision = business_decision.get('approval_decision')
        decision = normalize_decision(raw_decision)
        business_decision['approval_decision'] = decision

        unreadable = []
        if decision is None:
            unreadable.append('approval_decision')
            business_decision['parse_status'] = PARSE_UNPARSEABLE
        else:
            business_decision['parse_status'] = PARSE_OK

        # Strict token matching for approval_type
        raw_type = str(business_decision.get('approval_type', '')).strip().strip('.').upper()
        if decision == 'approve':
            if raw_type in {'STANDARD_TERMS', 'CONDITIONAL_APPROVAL', 'MANUAL_REVIEW'}:
                business_decision['approval_type'] = raw_type
            elif raw_type in {'STANDARD', 'STANDARD TERMS'}:
                business_decision['approval_type'] = 'STANDARD_TERMS'
            elif raw_type in {'CONDITIONAL', 'CONDITIONAL APPROVAL'}:
                business_decision['approval_type'] = 'CONDITIONAL_APPROVAL'
            elif raw_type in {'MANUAL', 'MANUAL REVIEW'}:
                business_decision['approval_type'] = 'MANUAL_REVIEW'
            else:
                business_decision['approval_type'] = None
                unreadable.append('approval_type')
        elif decision == 'deny':
            if raw_type in {'DENIAL', 'DENY'}:
                business_decision['approval_type'] = 'DENIAL'
            else:
                business_decision['approval_type'] = None
                unreadable.append('approval_type')
        else:
            business_decision['approval_type'] = None
            if raw_type:
                unreadable.append('approval_type')

        # Normalize interest_rate with template-bound range (P-7)
        rate_min, rate_max = 0.0, 40.0
        if isinstance(getattr(self, 'template', None), dict):
            rules = self.template.get('validation_rules', {})
            r_range = rules.get('interest_rate_range')
            if isinstance(r_range, (list, tuple)) and len(r_range) == 2:
                try:
                    rate_min, rate_max = float(r_range[0]), float(r_range[1])
                except (ValueError, TypeError):
                    pass
        raw_rate = business_decision.get('interest_rate')
        rate = self._safe_float(raw_rate)
        if rate is not None and not (rate_min <= rate <= rate_max):
            rate = None
        business_decision['interest_rate'] = rate
        if rate is None:
            unreadable.append('interest_rate')

        # Normalize confidence_probability
        raw_conf = business_decision.get('confidence_probability')
        conf = self._safe_int(raw_conf)
        business_decision['confidence_probability'] = conf
        if conf is None:
            unreadable.append('confidence_probability')

        # Validate/derive confidence_level
        raw_lvl = str(business_decision.get('confidence_level', '')).strip().strip('.').lower()
        if raw_lvl in ('low', 'medium', 'high'):
            business_decision['confidence_level'] = raw_lvl
        elif conf is not None:
            business_decision['confidence_level'] = ('low' if conf < 40 else 'medium' if conf < 80 else 'high')
        else:
            business_decision['confidence_level'] = None
            unreadable.append('confidence_level')

        business_decision['agent_name'] = "Business Decision"
        business_decision['loan_type'] = "personal_loan"

        # Safe weight normalization with provenance tracking
        agent_influence = business_decision.get('agent_influence', {})
        if not isinstance(agent_influence, dict):
            agent_influence = {}

        expected_keys = {'risk_manager_weight', 'regulatory_weight', 'data_science_weight', 'consumer_advocate_weight'}
        provided_weight_keys = {k for k in agent_influence.keys() if k not in ('primary_influence', 'basis_points', 'weights_status')}

        if provided_weight_keys != expected_keys:
            weights_status = 'imputed_cardinality'
            raw_weights = [0.25, 0.25, 0.25, 0.25]
            clean_weights = [0.25, 0.25, 0.25, 0.25]
            floored_units = [2500, 2500, 2500, 2500]
        else:
            raw_weights = [
                self._safe_float(agent_influence.get('risk_manager_weight')),
                self._safe_float(agent_influence.get('regulatory_weight')),
                self._safe_float(agent_influence.get('data_science_weight')),
                self._safe_float(agent_influence.get('consumer_advocate_weight'))
            ]
            clean_weights = []
            is_degenerate = False
            for w in raw_weights:
                try:
                    if w is not None and isinstance(w, (int, float)) and math.isfinite(w):
                        clean_weights.append(max(0.0, float(w)))
                    else:
                        clean_weights.append(0.0)
                        is_degenerate = True
                except (TypeError, ValueError, OverflowError):
                    clean_weights.append(0.0)
                    is_degenerate = True

            max_w = max(clean_weights) if clean_weights else 0.0

            if max_w > 0.0:
                scaled = [w / max_w for w in clean_weights]
                s_sum = sum(scaled)
                if s_sum > 1e-6 and math.isfinite(s_sum):
                    exact_units = [(sw / s_sum) * 10000.0 for sw in scaled]
                    floored_units = [int(math.floor(u)) for u in exact_units]
                    remainder = 10000 - sum(floored_units)
                    rank_order = sorted(range(len(exact_units)), key=lambda idx: exact_units[idx] - floored_units[idx], reverse=True)
                    for r_idx in range(remainder):
                        floored_units[rank_order[r_idx]] += 1
                    clean_weights = [round(fu / 10000.0, 4) for fu in floored_units]
                    weights_status = 'imputed_degenerate' if is_degenerate else 'model'
                else:
                    weights_status = 'imputed_degenerate'
                    clean_weights = [0.25, 0.25, 0.25, 0.25]
                    floored_units = [2500, 2500, 2500, 2500]
            else:
                weights_status = 'imputed_degenerate'
                clean_weights = [0.25, 0.25, 0.25, 0.25]
                floored_units = [2500, 2500, 2500, 2500]

        # Rebuild clean agent_influence with strictly allowed keys only
        business_decision['agent_influence'] = {
            'risk_manager_weight': clean_weights[0],
            'regulatory_weight': clean_weights[1],
            'data_science_weight': clean_weights[2],
            'consumer_advocate_weight': clean_weights[3],
            'primary_influence': str(agent_influence.get('primary_influence', 'None')),
            'weights_status': weights_status,
            'basis_points': {
                'risk_manager': floored_units[0],
                'regulatory': floored_units[1],
                'data_science': floored_units[2],
                'consumer_advocate': floored_units[3],
                'total': sum(floored_units)
            }
        }

        parse_detail_parts = []
        if unreadable:
            parse_detail_parts.append(f"unreadable({','.join(sorted(set(unreadable)))})")
        if weights_status != 'model':
            parse_detail_parts.append(f"weights({weights_status})")
        business_decision['parse_detail'] = '; '.join(parse_detail_parts) if parse_detail_parts else 'ok'

        return _sanitize_no_decimals(business_decision)

    def _create_error_response(self, error_message: str, raw_excerpt: str = "") -> dict:
        record = {
            "agent_name": "Business Decision",
            "loan_type": "personal_loan",
            "approval_decision": None,
            "approval_type": None,
            "interest_rate": None,
            "confidence_probability": None,
            "confidence_level": None,
            "parse_status": PARSE_ERROR,
            "parse_detail": error_message,
            "agent_influence": {
                "risk_manager_weight": 0.25,
                "regulatory_weight": 0.25,
                "data_science_weight": 0.25,
                "consumer_advocate_weight": 0.25,
                "primary_influence": "None",
                "basis_points": {
                    "risk_manager": 2500,
                    "regulatory": 2500,
                    "data_science": 2500,
                    "consumer_advocate": 2500,
                    "total": 10000
                }
            },
            "reasoning": {
                "synthesis_rationale": "Generation or parsing failed",
                "weight_justification": "Default equal weights applied",
                "risk_assessment": "Unassessed due to error"
            }
        }
        if raw_excerpt:
            record["raw_excerpt"] = raw_excerpt[:500]
            record["raw_response"] = raw_excerpt
        return record
