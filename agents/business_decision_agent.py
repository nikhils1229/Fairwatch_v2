import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from utils.async_client import AsyncLLMClient

logger = logging.getLogger(__name__)

class BusinessDecisionAgent(BaseAgent):
    """
    Agent that synthesizes final decisions from multiple agent recommendations.
    """
    def __init__(self, client: AsyncLLMClient):
        project_root = Path(__file__).parent.parent
        template_path = project_root / "templates" / "business_synthesis_template.json"
        persona_path = project_root / "prompts" / "business_decision_persona.txt"
        
        super().__init__(
            client, 
            "business_decision", 
            template_path=str(template_path),
            persona_path=str(persona_path)
        )

    def _build_system_prompt(self) -> str:
        schema = self.template.get('output_schema', {})
        instructions = self.template.get('synthesis_instructions', {})
        
        system_prompt = f"""{self.persona}

You MUST respond in valid JSON matching this exact schema:
{json.dumps(schema, indent=2)}

Synthesis Instructions:
- Task: {instructions.get('task')}
- Weighing Guidelines: {", ".join(instructions.get('weighing_guidelines', []))}
- Decision Logic: {", ".join(instructions.get('decision_logic', []))}

CRITICAL RULES:
- Output ONLY valid JSON.
- Include ALL fields from schema.
- Agent influence weights MUST sum to 1.0.
- interest_rate must be a float between 3.0 and 25.0.
- confidence_probability must be an integer 0-100.
- NEVER use null or omit fields.
"""
        return system_prompt

    def _heal_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Custom healing for business synthesis schema.
        """
        try:
            import json_repair as _jr
        except ImportError:
            _jr = None

        # 1. Basic cleaning
        text = re.sub(r'```json\s*', '', raw_text)
        text = re.sub(r'```', '', text)

        # 2. Syntax repair — B4 fix: mirror base_agent and prefer json_repair
        if _jr:
            repaired_json = _jr.repair_json(text)
        else:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                repaired_json = text[start:end+1]
            else:
                repaired_json = text

        try:
            parsed = json.loads(repaired_json)
        except Exception as e:
            logger.warning(f"JSON parsing failed for BusinessDecision: {e}")
            return self._create_error_response("Invalid JSON structure")

        if not isinstance(parsed, dict):
            return self._create_error_response("Response is not a JSON object")

        # 3. Semantic preservation and type enforcement
        parsed['agent_name'] = "Business Decision"
        parsed['loan_type'] = "personal_loan"

        # Decision normalization
        decision = str(parsed.get('approval_decision', 'deny')).lower()
        parsed['approval_decision'] = 'approve' if 'approve' in decision else 'deny'

        # B6 fix: normalise approval_type (was missing entirely)
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
        parsed['interest_rate'] = self._to_float(rate, default=15.0)
        parsed['interest_rate'] = max(3.0, min(25.0, parsed['interest_rate']))

        # Confidence enforcement
        prob = parsed.get('confidence_probability')
        parsed['confidence_probability'] = int(self._to_float(prob, default=50.0))
        parsed['confidence_probability'] = max(0, min(100, parsed['confidence_probability']))

        # B5 fix: derive confidence_level from confidence_probability (was missing)
        if parsed.get('confidence_level') not in ['low', 'medium', 'high']:
            cp = parsed['confidence_probability']
            if cp < 40:
                parsed['confidence_level'] = 'low'
            elif cp < 80:
                parsed['confidence_level'] = 'medium'
            else:
                parsed['confidence_level'] = 'high'

        # Weight normalization
        influence = parsed.get('agent_influence', {})
        if not isinstance(influence, dict):
            influence = {}

        weights = {
            'risk_manager_weight': self._to_float(influence.get('risk_manager_weight'), 0.25),
            'regulatory_weight': self._to_float(influence.get('regulatory_weight'), 0.25),
            'data_science_weight': self._to_float(influence.get('data_science_weight'), 0.25),
            'consumer_advocate_weight': self._to_float(influence.get('consumer_advocate_weight'), 0.25)
        }

        total_weight = sum(weights.values())
        if total_weight == 0:
            weights = {k: 0.25 for k in weights}
        else:
            weights = {k: v / total_weight for k, v in weights.items()}

        influence.update(weights)
        if 'primary_influence' not in influence:
            influence['primary_influence'] = "Risk Manager"
        if 'influence_explanation' not in influence:
            influence['influence_explanation'] = "Balanced weighting applied."

        parsed['agent_influence'] = influence

        # Reasoning block
        reasoning = parsed.get('reasoning', {})
        if not isinstance(reasoning, dict):
            reasoning = {}

        required_fields = [
            "approval_decision_reason",
            "approval_type_reason",
            "interest_rate_reason",
            "confidence_reason"
        ]
        for field in required_fields:
            if field not in reasoning:
                reasoning[field] = "Synthesized from agent inputs."

        parsed['reasoning'] = reasoning

        return parsed

    async def synthesize_decision(self, application_data: str, agent_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize final decision from agent recommendations.
        """
        formatted_recs = []
        for rec in agent_recommendations:
            name = rec.get('agent_name', 'Unknown Agent')
            decision = rec.get('approval_decision', 'N/A')
            app_type = rec.get('approval_type', 'N/A')
            rate = rec.get('interest_rate', 'N/A')
            conf = rec.get('confidence_probability', 'N/A')
            reasoning = rec.get('reasoning', {}).get('approval_decision_reason', 'N/A')
            
            rec_text = f"""
Agent: {name}
- Decision: {decision}
- Type: {app_type}
- Interest Rate: {rate}%
- Confidence: {conf}%
- Reasoning: {reasoning}
"""
            formatted_recs.append(rec_text.strip())
        
        recommendations_text = "\n\n".join(formatted_recs)
        
        prompt = f"""LOAN APPLICATION:
{application_data}

AGENT RECOMMENDATIONS:
{recommendations_text}

TASK:
Synthesize a final business decision considering all agent perspectives.
"""
        
        res = await self.client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            config=self.config,
            json_schema=self.template.get('output_schema')
        )
        
        if res.success:
            return self._heal_json(res.text)
            
        return self._create_error_response(f"Synthesis failed: {res.error_message}")

    async def evaluate_pairwise_comparison(self, context: str, current_prompt: str) -> Dict[str, Any]:
        return {"status": "not_implemented", "agent": self.agent_name}

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        return {
            "agent_name": "Business Decision",
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
            },
            "agent_influence": {
                "risk_manager_weight": 0.25,
                "regulatory_weight": 0.25,
                "data_science_weight": 0.25,
                "consumer_advocate_weight": 0.25,
                "primary_influence": "None",
                "influence_explanation": "Error occurred during synthesis."
            }
        }
