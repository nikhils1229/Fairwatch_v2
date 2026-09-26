"""
FairWatch V2 - Information Transmission Functions (phi)
Implements phi_full, phi_evidence, and phi_verdict to control upstream communication across multi-agent topologies.
"""

from typing import Dict, Any, Tuple
import json

def count_tokens(text: str) -> int:
    """Fast conservative token count approximation (whitespace + punctuation ratio)."""
    if not text:
        return 0
    # Standard 4 chars per token rule of thumb for English
    return max(1, len(text) // 4)

def phi_full(record: Dict[str, Any]) -> Tuple[str, int]:
    """
    phi_full: Transmits complete agent deliberation including all reasoning fields,
    the approval decision, approval type, interest rate, and confidence.
    """
    agent = record.get("agent_name", "Agent")
    decision = record.get("approval_decision", "unreadable")
    app_type = record.get("approval_type", "None")
    rate = record.get("interest_rate")
    rate_str = f"{rate}%" if rate is not None else "N/A"
    conf = record.get("confidence_probability")
    conf_str = f"{conf}%" if conf is not None else "N/A"
    conf_lvl = record.get("confidence_level", "N/A")
    
    reasoning = record.get("reasoning", {})
    if not isinstance(reasoning, dict):
        reasoning = {}
    
    lines = [
        f"=== {agent} Deliberation ===",
        f"Approval Decision: {decision}",
        f"Approval Type: {app_type}",
        f"Recommended Rate: {rate_str}",
        f"Confidence: {conf_str} ({conf_lvl})",
        "Reasoning:",
        f"  - Decision Rationale: {reasoning.get('approval_decision_reason', 'Not provided')}",
        f"  - Type Rationale: {reasoning.get('approval_type_reason', 'Not provided')}",
        f"  - Rate Rationale: {reasoning.get('interest_rate_reason', 'Not provided')}",
        f"  - Confidence Rationale: {reasoning.get('confidence_reason', 'Not provided')}",
    ]
    text = "\n".join(lines)
    return text, count_tokens(text)

import re

def sanitize_evidence_text(text: str) -> str:
    """Sanitizes text by redacting rates, percentages, and explicit verdict words without false positives."""
    if not text:
        return ""
    # 0. Ranges with percent/bps/rate units: "8.5-9.0%", "8.5 to 9.0%", "8-10 percent", "8.5 - 9.0%"
    s = re.sub(r'(?i)\b\d+(\.\d+)?\s*(?:-|–|—|\bto\b)\s*\d+(\.\d+)?\s*(?:%|percent(?:age)?\b|pct\b|bps\b|basis\s*points?\b)', '[RATE_REDACTED]', text)
    # 1. Individual percentages and basis points: 8.5%, 8.5 percent, 8.5 pct, 850 bps
    s = re.sub(r'(?i)\b\d+(\.\d+)?\s*(?:%|percent(?:age)?\b|pct\b|bps\b|basis\s*points?\b)', '[RATE_REDACTED]', s)
    # 2. Rate / APR / pricing context phrases: 'rate is 8.5', 'rate of 8.5', 'the rate is 8.5', 'priced at 8.5'
    s = re.sub(r'(?i)\b(?:(?:interest\s*)?rate|pricing|priced|apr)\s*(?:is|of|at|to|be|would\s*be|:|was|=)?\s*\d+(\.\d+)?\b', '[RATE_REDACTED]', s)
    # 3. Action verbs proposing numeric rates (with optional connectors like 'is'): 'proposing 8.5 as', 'the offer is 8.5', 'recommend 8.5'
    s = re.sub(r'(?i)\b(?:propos\w*|offer\w*|recommend\w*|suggest\w*|assign\w*|set\w*)\s*(?:is|of|at|to|be|would\s*be|:|was|=)?\s*(?:a\s+|an\s+|the\s+)?(?:rate\s+|apr\s+)?\d+(\.\d+)?\s*(?:as|for|in|compensation|to|depending)?\b', '[RATE_REDACTED]', s)
    # 4. Standalone numbers in typical loan rate range (1-40) preceded by at/of/around, protecting non-rate units (years, jobs, etc.)
    s = re.sub(r'(?i)\b(?:at|of|around)\s+(?:[1-9]|[1-3]\d|40)(?:\.\d{1,2})?\b(?!\s*(?:years?|months?|days?|jobs?|employers?|accounts?|times?|trades?|inquir\w*|dependents?))', '[RATE_REDACTED]', s)
    # 5. Redact explicit approval/denial/disapproval verdict terms (avoiding words like Denver or density)
    s = re.sub(r'(?i)\b(disapprov\w*|approv\w*|den(y|ied|ial|ials|ies|ying)\b|declin\w*|reject\w*)\b', '[VERDICT_REDACTED]', s)
    return s

def phi_evidence(record: Dict[str, Any]) -> Tuple[str, int]:
    """
    phi_evidence: Transmits only analytical reasoning and observations,
    deliberately stripping out the final verdict, rate, and approval decision to test causal influence.
    """
    agent = record.get("agent_name", "Agent")
    reasoning = record.get("reasoning", {})
    if not isinstance(reasoning, dict):
        reasoning = {}
    
    lines = [
        f"=== {agent} Evidence & Analysis ===",
        f"  - Risk Factors: {sanitize_evidence_text(str(reasoning.get('approval_decision_reason', 'Not provided')))}",
        f"  - Financial Analysis: {sanitize_evidence_text(str(reasoning.get('interest_rate_reason', 'Not provided')))}",
        f"  - Profile Assessment: {sanitize_evidence_text(str(reasoning.get('confidence_reason', 'Not provided')))}",
    ]
    text = "\n".join(lines)
    return text, count_tokens(text)

def phi_verdict(record: Dict[str, Any]) -> Tuple[str, int]:
    """
    phi_verdict: Transmits only the bare numerical/categorical decision,
    stripping out all qualitative narrative reasoning.
    """
    agent = record.get("agent_name", "Agent")
    decision = record.get("approval_decision", "unreadable")
    app_type = record.get("approval_type", "None")
    rate = record.get("interest_rate")
    rate_str = f"{rate}%" if rate is not None else "N/A"
    
    lines = [
        f"=== {agent} Verdict ===",
        f"Decision: {decision}",
        f"Type: {app_type}",
        f"Rate: {rate_str}",
    ]
    text = "\n".join(lines)
    return text, count_tokens(text)
