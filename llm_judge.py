# # llm_judge.py
# import json
# import re
# from typing import Dict, List, Optional
#
# from transformers import pipeline
#
# # Strict: find a JSON object that actually has "label": "<one of 4>"
# _JSON_OBJ_RE = re.compile(
#     r'\{[^{}]*"label"\s*:\s*"(?:satisfied|partially_satisfied|non_existent|garbage)"[^{}]*\}',
#     re.IGNORECASE | re.DOTALL,
# )
#
# INSTRUCTIONS = """
# Please compare the compliance 'text with the policy 'best match'  If the compliance text is not a statement or rule, mark as garbage.
# if the compliance text is clearly satisfied by the policy 'best match' please mark as satisfied.
# If there is any relation between the compliance text and the policy best match please mark ast partially_satisfied.
# If the compliance text is a valid statement or rule but the best match is unrelated, mark as non_existent
#
# Reply with ONLY a single-line JSON object:
# {"label":"<satisfied|partially_satisfied|non_existent|garbage>","rationale":"<one short, specific sentence>"}
#
# # Rules:
# # - "garbage": the compliance text is unrelated or just noise (generic words without substance).
# # - "satisfied": clearly fulfills the requirement.
# # - "partially_satisfied": addresses the idea but is weaker, missing a key element, or has looser scope/threshold.
# # - "non_existent": does not cover the requirement.
# # Do not include any extra text, code fences, or explanations.
# # """
#
# PROMPT_TEMPLATE = """{instructions}
#
# REQUIREMENT:
# {requirement}
#
# BEST POLICY MATCH:
# {best}
#
# TOP ALTERNATIVES:
# {alts}
#
# JSON:
# """
#
# def _extract_json(text: str) -> Optional[str]:
#     matches = list(_JSON_OBJ_RE.finditer(text))
#     if not matches:
#         return None
#     # take the LAST json-looking object to avoid the model echoing earlier examples
#     return matches[-1].group(0)
#
# def _normalize_label(x: str) -> str:
#     x = x.strip().lower()
#     if x in {"satisfied", "partially_satisfied", "non_existent", "garbage"}:
#         return x
#     # normalize variants
#     if x.replace("-", "_") == "non_existent" or x.replace(" ", "_") == "non_existent":
#         return "non_existent"
#     return "non_existent"
#
# class LLMJudge:
#     def __init__(self, model_name: str = "google/flan-t5-base", device: int = -1, hf_token: Optional[str] = None, max_new_tokens: int = 192):
#         self.pipe = pipeline(
#             task="text2text-generation",
#             model=model_name,
#             device=device,
#             use_auth_token=hf_token if hf_token else None,
#         )
#         self.max_new_tokens = max_new_tokens
#
#     def assess(self, requirement: str, best_policy: str, alt_snippets: Optional[List[str]] = None) -> Dict[str, str]:
#         alts_txt = ""
#         if alt_snippets:
#             for s in alt_snippets[:2]:
#                 alts_txt += f"- {s}\n"
#
#         prompt = PROMPT_TEMPLATE.format(
#             instructions=INSTRUCTIONS.strip(),
#             requirement=requirement.strip(),
#             best=(best_policy or "(none)").strip(),
#             alts=(alts_txt.strip() or "(none)"),
#         )
#
#         out = self.pipe(
#             prompt,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=False,   # deterministic
#             num_beams=1,
#             return_full_text=False,
#         )[0]["generated_text"].strip()
#
#         js = _extract_json(out)
#         if js:
#             try:
#                 obj = json.loads(js)
#                 label = _normalize_label(obj.get("label", "non_existent"))
#                 rationale = obj.get("rationale", "").strip()
#                 return {"label": label, "rationale": rationale, "_parsed": True}
#             except Exception:
#                 pass
#
#         # final fallback: DO NOT GUESS a label, just report unparsed
#         return {
#             "label": "non_existent",
#             "rationale": "LLM output was not valid JSON; ignoring.",
#             "_parsed": False,
#         }


# llm_judge.py (debuggable)
import json
import os
import re
import time
import uuid
import logging
from typing import Dict, List, Optional

from transformers import pipeline, AutoConfig

# ----------------------------
# Logging config (module-level)
# ----------------------------
LOGGER_NAME = "llm_judge"
logger = logging.getLogger(LOGGER_NAME)
# If the app hasn't configured handlers, add a basic one.
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
# Level is controlled by env; default INFO (set DEBUG for chatty)
logger.setLevel(os.getenv("LLM_JUDGE_LOGLEVEL", "INFO").upper())

# Truncation for log safety
MAX_LOG_CHARS = int(os.getenv("LLM_JUDGE_MAX_LOG_CHARS", "800"))  # per field
def _clip(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= MAX_LOG_CHARS else s[:MAX_LOG_CHARS] + f"... [truncated {len(s)-MAX_LOG_CHARS} chars]"

# Strict: find a JSON object that actually has "label": "<one of 4>"
_JSON_OBJ_RE = re.compile(
    r'\{[^{}]*"label"\s*:\s*"(?:satisfied|partially_satisfied|non_existent|garbage)"[^{}]*\}',
    re.IGNORECASE | re.DOTALL,
)

INSTRUCTIONS = """
Compare the COMPLIANCE requirement against the POLICY best match.
- If the compliance text is not a clear requirement/statement, label "garbage".
- If the policy clearly satisfies the requirement, label "satisfied".
- If the policy is related but incomplete/weaker/missing a key element, label "partially_satisfied".
- If the requirement is valid but the policy is unrelated, label "non_existent".

Reply with ONLY one JSON object on a single line:
{"label":"<satisfied|partially_satisfied|non_existent|garbage>","rationale":"<one short, specific sentence>"}
"""

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
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: int = -1,
        hf_token: Optional[str] = None,
        max_new_tokens: int = 192,
    ):
        logger.info(f"[INIT] Creating LLMJudge for model={model_name}, device={device}")

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))
        self.task = "text2text-generation" if self.is_encoder_decoder else "text-generation"

        logger.info(f"[INIT] Detected arch: {'encoder-decoder' if self.is_encoder_decoder else 'decoder-only'}; using task={self.task}")

        # Build pipeline
        self.pipe = pipeline(
            task=self.task,
            model=model_name,
            device=device,
            # prefer token=... on newer versions; keep use_auth_token for back-compat
            use_auth_token=hf_token if hf_token else None,
        )
        self.max_new_tokens = max_new_tokens

        # For decoder-only models, avoid pad warnings in generate()
        # (we avoid touching tokenizer directly here; pipeline handles it in most cases)
        logger.debug(f"[INIT] max_new_tokens={self.max_new_tokens}")

    def assess(
        self,
        requirement: str,
        best_policy: str,
        alt_snippets: Optional[List[str]] = None
    ) -> Dict[str, str]:
        req_id = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()

        alts_txt = ""
        if alt_snippets:
            for s in alt_snippets[:2]:
                alts_txt += f"- {s}\n"

        prompt = PROMPT_TEMPLATE.format(
            instructions=INSTRUCTIONS.strip(),
            requirement=(requirement or "").strip(),
            best=(best_policy or "(none)").strip(),
            alts=(alts_txt.strip() or "(none)"),
        )

        logger.info(f"[{req_id}] LLMJudge.assess called")
        logger.debug(f"[{req_id}] requirement={_clip(requirement)}")
        logger.debug(f"[{req_id}] best_policy={_clip(best_policy)}")
        if alts_txt:
            logger.debug(f"[{req_id}] alt_snippets={_clip(alts_txt)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[{req_id}] prompt=\n{_clip(prompt)}")

        try:
            # IMPORTANT: do not pass return_full_text
            out_obj = self.pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )[0]
            # Both text-generation and text2text-generation return "generated_text"
            out = (out_obj.get("generated_text") or "").strip()
        except Exception as e:
            logger.exception(f"[{req_id}] pipeline call failed: {e}")
            return {
                "label": "non_existent",
                "rationale": f"LLM pipeline error: {e}",
                "_parsed": False,
            }

        logger.debug(f"[{req_id}] raw_output={_clip(out)}")

        js = _extract_json(out)
        if js:
            logger.debug(f"[{req_id}] extracted_json={_clip(js)}")
            try:
                obj = json.loads(js)
                label = _normalize_label(obj.get("label", "non_existent"))
                rationale = (obj.get("rationale") or "").strip()
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"[{req_id}] parsed label={label}; latency={dt:.1f} ms")
                if rationale:
                    logger.debug(f"[{req_id}] rationale={_clip(rationale)}")
                return {"label": label, "rationale": rationale, "_parsed": True}
            except Exception as e:
                logger.exception(f"[{req_id}] json.loads failed: {e}")

        dt = (time.perf_counter() - t0) * 1000
        logger.warning(f"[{req_id}] no valid JSON found; latency={dt:.1f} ms")
        return {
            "label": "non_existent",
            "rationale": "LLM output was not valid JSON; ignoring.",
            "_parsed": False,
        }