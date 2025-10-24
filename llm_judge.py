# llm_judge.py  (NEW FILE)
import json
import re
from typing import Dict, List, Optional

from transformers import pipeline

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)

INSTRUCTIONS = """You are a strict compliance analyst. Given a requirement from a regulation and the closest matching company policy text, decide how well the policy satisfies the requirement.

Return ONLY a compact JSON object with:
- "label": one of ["satisfied","partially_satisfied","non_existent","garbage"]
- "rationale": a single, specific sentence why.

Rules:
- "garbage": the policy text is unrelated (generic overlap like 'policy' or 'data' without substance).
- "satisfied": the policy clearly fulfills the requirement.
- "partially_satisfied": policy addresses the idea but is weaker, missing a key element, or has looser scope/threshold (e.g., 48h vs 24h).
- "non_existent": the policy text doesn't cover the requirement.

Be conservative and use the requirement exactly as written. Keep the rationale short.
"""

PROMPT_TEMPLATE = """{instructions}

Requirement:
{requirement}

Best policy match:
{best}

Top alternative snippets:
{alts}
"""

def _extract_json(text: str) -> Optional[str]:
    m = _JSON_RE.search(text)
    return m.group(0) if m else None

class LLMJudge:
    def __init__(self, model_name: str = "google/flan-t5-base", device: int = -1, hf_token: Optional[str] = None, max_new_tokens: int = 192):
        # text2text works well with FLAN-T5 and is CPU-friendly
        self.pipe = pipeline(
            task="text2text-generation",
            model=model_name,
            device=device,
            use_auth_token=hf_token if hf_token else None
        )
        self.max_new_tokens = max_new_tokens

    def assess(self, requirement: str, best_policy: str, alt_snippets: Optional[List[str]] = None) -> Dict[str, str]:
        alts_txt = ""
        if alt_snippets:
            # limit to two short alts
            for s in alt_snippets[:2]:
                alts_txt += f"- {s}\n"

        prompt = PROMPT_TEMPLATE.format(
            instructions=INSTRUCTIONS.strip(),
            requirement=requirement.strip(),
            best=best_policy.strip() if best_policy else "(none)",
            alts=alts_txt.strip() or "(none)"
        )

        out = self.pipe(prompt, max_new_tokens=self.max_new_tokens)[0]["generated_text"].strip()
        js = _extract_json(out)
        if js:
            try:
                return json.loads(js)
            except Exception:
                pass
        # Fallback if model didn't return clean JSON
        return {"label": "non_existent", "rationale": "Could not parse LLM output; defaulting to non_existent."}
