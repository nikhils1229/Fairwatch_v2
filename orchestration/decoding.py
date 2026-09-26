"""
FairWatch V2 - Sampling and Decoding Policy
Enforces immutable sampling contracts, parameter bounds, and policy fingerprinting.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import sys
from typing import Dict, Any, Union

assert sys.version_info >= (3, 10), "FairWatch V2 requires Python >= 3.10 for dataclass(slots=True)"

_FIELDS = ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")

@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    policy_id: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_tokens: int

    def fingerprint(self) -> str:
        """Returns 16-character sha256 fingerprint. Seed does NOT enter fingerprint."""
        payload = {k: getattr(self, k) for k in _FIELDS}
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


def validate_sampling(p: SamplingPolicy) -> SamplingPolicy:
    """Validates sampling parameters with strict bounds and vLLM constraints."""
    def chk(name: str, v: Any, lo: float, hi: float, integral: bool):
        if isinstance(v, bool):
            raise TypeError(f"{name}: bool is not numeric")
        if v is None:
            raise TypeError(f"{name}: cannot be None")
        if integral:
            if not isinstance(v, int):
                raise TypeError(f"{name}: must be int")
        else:
            if not isinstance(v, (int, float)):
                raise TypeError(f"{name}: must be numeric")
            if not math.isfinite(float(v)):
                raise ValueError(f"{name}: non-finite value {v}")
        if not (lo <= v <= hi):
            raise ValueError(f"{name}={v} outside allowed range [{lo}, {hi}]")

    chk("temperature", p.temperature, 0.0, 2.0, False)
    chk("top_p", p.top_p, 0.0, 1.0, False)
    chk("top_k", p.top_k, -1, 10**6, True)
    if p.top_k == 0:
        raise ValueError("top_k=0 is invalid for vLLM: must be -1 (disabled) or >= 1")
    chk("repetition_penalty", p.repetition_penalty, 0.5, 2.0, False)
    chk("max_tokens", p.max_tokens, 1, 8192, True)
    return p


def normalize_sampling(config: Union[SamplingPolicy, Dict[str, Any]]) -> SamplingPolicy:
    """Converts a dict or SamplingPolicy to a validated SamplingPolicy. Fails loudly on partial/unknown configs."""
    if isinstance(config, SamplingPolicy):
        return validate_sampling(config)
    if isinstance(config, dict):
        unknown = set(config) - set(_FIELDS) - {"policy_id"}
        if unknown:
            raise ValueError(f"Unknown sampling parameters: {sorted(unknown)}")
        missing = set(_FIELDS) - set(config)
        if missing:
            raise ValueError(f"Missing required sampling parameters: {sorted(missing)}")
        policy_id = str(config.get("policy_id", "adhoc"))
        return validate_sampling(SamplingPolicy(
            policy_id=policy_id,
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            top_k=int(config["top_k"]),
            repetition_penalty=float(config["repetition_penalty"]),
            max_tokens=int(config["max_tokens"])
        ))
    raise TypeError(f"Unusable generation config: {type(config)!r}")


# Canonical Pinned Policies
PRIMARY = validate_sampling(SamplingPolicy(
    policy_id="primary_greedy",
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.0,
    max_tokens=1024
))

TEMP_020 = validate_sampling(SamplingPolicy(
    policy_id="temp_020",
    temperature=0.2,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.0,
    max_tokens=1024
))

TEMP_070 = validate_sampling(SamplingPolicy(
    policy_id="temp_070",
    temperature=0.7,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.0,
    max_tokens=1024
))

POLICIES: Dict[str, SamplingPolicy] = {
    "primary": PRIMARY,
    "primary_greedy": PRIMARY,
    "temp_020": TEMP_020,
    "temp_070": TEMP_070,
}
