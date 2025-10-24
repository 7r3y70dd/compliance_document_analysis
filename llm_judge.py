# llm_judge.py
import json
import re
from typing import Dict, List, Optional

from transformers import pipeline

# Strict: find a JSON object that actually has "label": "<one of 4>"
_JSON_OBJ_RE = re.compile(
    r'\{[^{}]*"label"\s*:\s*"(?:satisfied|partially_satisfied|non_existent|garbage)"[^{}]*\}',
    re.IGNORECASE | re.DOTALL,
)

INSTRUCTIONS = """
Please compare the compliance 'text with the policy 'best match'  If the compliance text is not a statement or rule, mark as garbage.  
if the compliance text is clearly satisfied by the policy 'best match' please mark as satisfied.  
If there is any relation between the compliance text and the policy best match please mark ast partially_satisfied.
If the compliance text is a valid statement or rule but the best match is unrelated, mark as non_existent

Reply with ONLY a single-line JSON object:
{"label":"<satisfied|partially_satisfied|non_existent|garbage>","rationale":"<one short, specific sentence>"}

# Rules:
# - "garbage": the compliance text is unrelated or just noise (generic words without substance).
# - "satisfied": clearly fulfills the requirement.
# - "partially_satisfied": addresses the idea but is weaker, missing a key element, or has looser scope/threshold.
# - "non_existent": does not cover the requirement.
# Do not include any extra text, code fences, or explanations.
# """

PROMPT_TEMPLATE = """{instructions}

REQUIREMENT:
{requirement}

BEST POLICY MATCH:
{best}

TOP ALTERNATIVES:
{alts}

JSON:
"""

def _extract_json(text: str) -> Optional[str]:
    matches = list(_JSON_OBJ_RE.finditer(text))
    if not matches:
        return None
    # take the LAST json-looking object to avoid the model echoing earlier examples
    return matches[-1].group(0)

def _normalize_label(x: str) -> str:
    x = x.strip().lower()
    if x in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
        return x
    # normalize variants
    if x.replace("-", "_") == "non_existent" or x.replace(" ", "_") == "non_existent":
        return "non_existent"
    return "non_existent"

class LLMJudge:
    def __init__(self, model_name: str = "google/flan-t5-base", device: int = -1, hf_token: Optional[str] = None, max_new_tokens: int = 192):
        self.pipe = pipeline(
            task="text2text-generation",
            model=model_name,
            device=device,
            use_auth_token=hf_token if hf_token else None,
        )
        self.max_new_tokens = max_new_tokens

    def assess(self, requirement: str, best_policy: str, alt_snippets: Optional[List[str]] = None) -> Dict[str, str]:
        alts_txt = ""
        if alt_snippets:
            for s in alt_snippets[:2]:
                alts_txt += f"- {s}\n"

        prompt = PROMPT_TEMPLATE.format(
            instructions=INSTRUCTIONS.strip(),
            requirement=requirement.strip(),
            best=(best_policy or "(none)").strip(),
            alts=(alts_txt.strip() or "(none)"),
        )

        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,   # deterministic
            num_beams=1,
            return_full_text=False,
        )[0]["generated_text"].strip()

        js = _extract_json(out)
        if js:
            try:
                obj = json.loads(js)
                label = _normalize_label(obj.get("label", "non_existent"))
                rationale = obj.get("rationale", "").strip()
                return {"label": label, "rationale": rationale, "_parsed": True}
            except Exception:
                pass

        # final fallback: DO NOT GUESS a label, just report unparsed
        return {
            "label": "non_existent",
            "rationale": "LLM output was not valid JSON; ignoring.",
            "_parsed": False,
        }
