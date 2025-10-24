# comparator.py
"""
Numeric comparator to catch stricter/looser requirements, e.g., 24h vs 48h.
Downgrades when the policy is weaker than the compliance clause.
"""
import re
from typing import Optional, Tuple

_TIME_RE = re.compile(
    r"(?:(within|no\s+later\s+than|not\s+later\s+than|under|less\s+than|at\s+least|min(?:imum)?|not\s+less\s+than)\s+)"
    r"(\d{1,4})\s*"
    r"(minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|w)",
    re.I
)

_UNIT_TO_MIN = {
    'm': 1, 'min': 1, 'mins': 1, 'minute': 1, 'minutes': 1,
    'h': 60, 'hr': 60, 'hrs': 60, 'hour': 60, 'hours': 60,
    'd': 1440, 'day': 1440, 'days': 1440,
    'w': 10080, 'week': 10080, 'weeks': 10080,
}

_MAX_CUES = {"within", "no later than", "not later than", "under", "less than"}   # ≤ deadline
_MIN_CUES = {"at least", "min", "minimum", "not less than"}                       # ≥ minimum

def _to_minutes(value: int, unit: str) -> int:
    u = unit.lower()
    u = {'hr': 'h', 'hrs': 'h'}.get(u, u)
    u = {'minute': 'minutes', 'hour': 'hours', 'day': 'days', 'week': 'weeks'}.get(u, u)
    return int(value) * _UNIT_TO_MIN.get(u, 1)

def _extract_bound(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (max_minutes, min_minutes) from the text."""
    maxs, mins = [], []
    for m in _TIME_RE.finditer(text):
        cue_raw, val, unit = m.groups()
        cue = cue_raw.lower()
        val_min = _to_minutes(int(val), unit)
        if any(cue.startswith(c) for c in _MAX_CUES):
            maxs.append(val_min)
        elif any(cue.startswith(c) for c in _MIN_CUES):
            mins.append(val_min)
    return (min(maxs) if maxs else None, max(mins) if mins else None)

def numeric_downgrade(compliance: str, policy: str) -> Tuple[bool, str]:
    """
    Return (downgrade, rationale).
    If compliance is ≤C and policy is ≤P, downgrade when P > C (policy looser).
    If compliance is ≥C and policy is ≥P, downgrade when P < C (policy looser).
    Mixed types are ignored.
    """
    c_max, c_min = _extract_bound(compliance)
    p_max, p_min = _extract_bound(policy)

    if c_max is not None and p_max is not None:
        if p_max > c_max:
            return True, (
                f"Numeric mismatch: compliance requires action within {c_max} minutes, "
                f"but policy allows {p_max} minutes."
            )
        return False, ""

    if c_min is not None and p_min is not None:
        if p_min < c_min:
            return True, (
                f"Numeric mismatch: compliance requires at least {c_min} minutes, "
                f"but policy guarantees only {p_min} minutes."
            )
        return False, ""

    return False, ""
