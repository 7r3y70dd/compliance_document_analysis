# rewriter.py
import json
import re
from typing import Optional, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)

# -------- JSON extractors --------
_EDIT_JSON_RE = re.compile(
    r'\{[^{}]*"original"\s*:\s*".*?"[^{}]*"rewritten"\s*:\s*".*?"[^{}]*"status"\s*:\s*"(?:unchanged|rewritten)"[^{}]*\}',
    re.DOTALL,
)

def _extract_edit_json(s: str) -> Optional[dict]:
    m = list(_EDIT_JSON_RE.finditer(s or ""))
    if not m:
        return None
    try:
        return json.loads(m[-1].group(0))
    except Exception:
        return None


# -------- Prompts --------
EDIT_INSTRUCTIONS = """
You are a policy editor. You ONLY rewrite the provided POLICY_CLAUSE to fully satisfy the COMPLIANCE_REQUIREMENT.

Rules:
- Keep the style, tone, and formatting consistent with the original clause.
- Be concise; do not add new sections or headings.
- Do not change numbering outside the clause; preserve bullets/section labels.
- If the clause already fully satisfies the requirement, return it unchanged.

Return ONLY one JSON object on a single line:
{"original":"<exact POLICY_CLAUSE>",
 "rewritten":"<improved clause, similar length and style>",
 "status":"<unchanged|rewritten>",
 "notes":"<very brief reason, <=12 words>"}
""".strip()

EDIT_PROMPT = """{instructions}

COMPLIANCE_REQUIREMENT:
{requirement}

CURRENT POLICY_CLAUSE (verbatim from the document):
{policy_clause}

RATIONALE (why the clause is not fully satisfied):
{rationale}

JSON:
"""

NEW_CLAUSE_INSTRUCTIONS = """
You are a policy writer. Convert the COMPLIANCE_REQUIREMENT into a single,
concise policy clause that would appear in a company policy.

Rules:
- Clear, prescriptive, and auditable language (“Company shall…”, “We must…”).
- Keep tone and length similar to existing clauses.
- Do not add headings or numbering; return only the clause sentence(s).
""".strip()

NEW_CLAUSE_PROMPT = """{instructions}

COMPLIANCE_REQUIREMENT:
{requirement}

Return only the clause text, no headings, no numbering:
"""


# rewriter.py (inside class EditorRewriter)

from transformers import AutoConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM

class EditorRewriter:
    """
    Small editor that: (a) proposes precise rewrites as strict JSON,
    (b) drafts entirely new clauses from requirements when coverage is missing.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: int = -1,
        max_new_tokens: int = 128,
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        # Decide task from config (no loading big weights yet)
        cfg = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token or None)
        # Heuristic: T5/MT5/mBART/etc. are encoder-decoder => text2text; most others => text-generation
        seq2seq_families = {
            "t5", "mt5", "mbart", "bart", "marian", "pegasus", "longt5", "umt5", "blenderbot", "prophetnet"
        }
        model_type = (getattr(cfg, "model_type", "") or "").lower()
        task = "text2text-generation" if model_type in seq2seq_families else "text-generation"

        # Try to enable 4-bit only when actually possible
        use_4bit = False
        if load_in_4bit and self.device != -1:
            try:
                import bitsandbytes as _  # noqa: F401
                use_4bit = True
            except Exception:
                use_4bit = False  # silently fall back

        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token or None)

        # Simple path: let pipeline load the model without device_map (so no accelerate needed)
        # If you *really* want 4-bit, we load the model manually with bitsandbytes; else use pipeline with name.
        if task == "text-generation":
            if use_4bit:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    use_auth_token=hf_token or None,
                    load_in_4bit=True,
                    torch_dtype=torch.float16,
                )
                self.pipe = pipeline(task, model=model, tokenizer=tok, device=device)
            else:
                # Let pipeline handle weights; no device_map used, so no accelerate required
                self.pipe = pipeline(task, model=model_name, tokenizer=tok, device=device, use_auth_token=hf_token or None)
        else:  # text2text-generation
            if use_4bit:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    use_auth_token=hf_token or None,
                    load_in_4bit=True,
                    torch_dtype=torch.float16,
                )
                self.pipe = pipeline(task, model=model, tokenizer=tok, device=device)
            else:
                self.pipe = pipeline(task, model=model_name, tokenizer=tok, device=device, use_auth_token=hf_token or None)

        self.is_text_generation = (self.pipe.task == "text-generation")


    # ---- Rewrite an existing clause (strict JSON) ----
    def propose_edit(self, requirement: str, policy_clause: str, rationale: str) -> Dict[str, str]:
        prompt = EDIT_PROMPT.format(
            instructions=EDIT_INSTRUCTIONS,
            requirement=(requirement or "").strip(),
            policy_clause=(policy_clause or "(none)").strip(),
            rationale=(rationale or "Make it fully satisfy the requirement.").strip(),
        )

        if self.is_text_generation:
            out = self.pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                return_full_text=False,
            )[0]["generated_text"].strip()
        else:
            out = self.pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )[0]["generated_text"].strip()

        obj = _extract_edit_json(out) or {}
        original = (obj.get("original") or policy_clause or "").strip()
        rewritten = (obj.get("rewritten") or policy_clause or "").strip()
        status = (obj.get("status") or "unchanged").strip().lower()
        notes = (obj.get("notes") or "").strip()
        if status not in ("unchanged", "rewritten"):
            status = "rewritten" if rewritten and rewritten != original else "unchanged"
        return {"original": original, "rewritten": rewritten, "status": status, "notes": notes}

    # ---- Draft a new clause from a requirement ----
    def draft_new_clause(self, requirement: str) -> str:
        prompt = NEW_CLAUSE_PROMPT.format(
            instructions=NEW_CLAUSE_INSTRUCTIONS,
            requirement=(requirement or "").strip(),
        )
        if self.is_text_generation:
            out = self.pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                return_full_text=False,
            )[0]["generated_text"].strip()
        else:
            out = self.pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )[0]["generated_text"].strip()

        return " ".join(out.split())
