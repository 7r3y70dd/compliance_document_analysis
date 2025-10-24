# Minimal NLI Judge (Hugging Face MNLI) — Non‑breaking Patch
# ----------------------------------------------------------
# This patch adds a small Natural Language Inference (NLI) judge that
# evaluates DIRECTIONAL coverage: whether the POLICY clause (premise)
# ENTAILS the COMPLIANCE clause (hypothesis).
#
# Usage modes (feature‑flagged):
# 1) Annotate‑only (default): adds an NLI verdict into `rationale` without
#    changing your existing labels. Enable with USE_NLI_JUDGE=1 (default
#    NLI_ANNOTATE_ONLY=1).
# 2) Replace/Constrain: optionally let NLI adjust the label toward the
#    *stricter* outcome (never upgrades). Set NLI_ANNOTATE_ONLY=0.
# ----------------------------------------------------------

# ================================
# file: nli_judge.py
# ================================
from __future__ import annotations
from typing import Dict, Tuple
from transformers import pipeline

from settings import settings

_LABELS = ("entailment", "neutral", "contradiction")

class NLIJudge:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.nli_model_name
        # text-classification pipeline supports pair input via dict
        self.pipe = pipeline(
            "text-classification",
            model=self.model_name,
            return_all_scores=True,
            truncation=True,
        )

    def scores(self, policy_clause: str, compliance_clause: str) -> Dict[str, float]:
        """Return probabilities for entailment/neutral/contradiction."""
        out = self.pipe({"text": policy_clause, "text_pair": compliance_clause})[0]
        bylabel = {s["label"].lower(): float(s["score"]) for s in out}
        # some models return labels like CONTRADICTION/NEUTRAL/ENTAILMENT
        return {k: bylabel.get(k, 0.0) for k in _LABELS}

    def classify(self, policy_clause: str, compliance_clause: str) -> Tuple[str, Dict[str, float]]:
        s = self.scores(policy_clause, compliance_clause)
        ent, neu, con = s["entailment"], s["neutral"], s["contradiction"]
        # Minimal heuristic thresholds (tunable via env)
        sat = settings.nli_satisfied_floor  # default 0.70
        par = settings.nli_partial_floor    # default 0.45
        if ent >= sat:
            label = "satisfied"
        elif ent >= par or neu >= 0.50:
            label = "partially_satisfied"
        else:
            # strong contradiction or very low entailment
            label = "non_existent"
        return label, s


def stricter(existing: str, proposed: str) -> str:
    """Return the stricter label (non_existent < partial < satisfied)."""
    rank = {"non_existent": 0, "partially_satisfied": 1, "satisfied": 2}
    return existing if rank[existing] <= rank[proposed] else proposed


