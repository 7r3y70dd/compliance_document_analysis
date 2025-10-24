from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

from settings import settings

class ClauseMatcher:
    def __init__(self):
        # Load embedding model once.
        self.embed = SentenceTransformer(settings.embedding_model_name)
        self._rationale = None
        self._judge = None

        # NEW: optional cross-encoder reranker
        self.reranker = None
        if settings.use_cross_encoder:
            self.reranker = CrossEncoder(settings.cross_encoder_model_name, max_length=512)

    # --- OLD ENCODING (kept for easy revert) ---
    # def encode(self, texts: List[str]) -> np.ndarray:
    #     return np.array(self.embed.encode(texts, normalize_embeddings=True))

    def encode(self, texts: List[str], kind: str = "passage") -> np.ndarray:
        """
        Encode texts with normalization. If using E5-family models, apply the required
        prefixes: 'query: ' for compliance clauses and 'passage: ' for policy clauses.
        kind: 'query' or 'passage'
        """
        to_encode = texts
        if settings.use_e5_prefixes or "e5" in settings.embedding_model_name.lower():
            prefix = "query: " if kind == "query" else "passage: "
            to_encode = [prefix + t for t in texts]
        return np.array(self.embed.encode(to_encode, normalize_embeddings=True))

    @staticmethod
    def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # A: mxd, B: nxd, both normalized -> cosine = dot
        return A @ B.T

    def best_matches(
        self,
        compliance_clauses: List[str],
        policy_clauses: List[str],
        top_k: int = 3,
    ) -> List[Tuple[int, List[Tuple[int, float]]]]:
        """Return, for each compliance clause i, a list of (policy_index, score).
        First use bi-encoder embeddings to shortlist; optionally rerank with a cross-encoder.
        """
        # Encode with appropriate prefixes
        c_emb = self.encode(compliance_clauses, kind="query")
        p_emb = self.encode(policy_clauses, kind="passage")
        sims = self.cosine_sim_matrix(c_emb, p_emb)

        results: List[Tuple[int, List[Tuple[int, float]]]] = []
        # If reranking, take a larger pool to re-score
        pool_k = max(top_k, settings.cross_encoder_top_k) if self.reranker else top_k

        for i in range(sims.shape[0]):
            row = sims[i]
            idx = np.argpartition(-row, range(min(pool_k, row.shape[0])))[:pool_k]
            idx = idx[np.argsort(-row[idx])]  # sort by similarity desc

            if self.reranker:
                # Build pairs for cross-encoder: (query, passage)
                pairs = [(compliance_clauses[i], policy_clauses[int(j)]) for j in idx]
                ce_scores = self.reranker.predict(pairs)
                # Sort by CE score descending
                order = np.argsort(-ce_scores)
                idx = idx[order]
                scores = ce_scores[order]
                # Use CE scores for ranking; fall back to bi-encoder sims if needed
                ranked = [(int(j), float(scores[k])) for k, j in enumerate(idx[:top_k])]
            else:
                ranked = [(int(j), float(row[j])) for j in idx[:top_k]]

            results.append((i, ranked))
        return results

    def label_from_similarity(self, sim: float) -> str:
        if sim >= settings.satisfied_threshold:
            return "satisfied"
        if sim >= settings.partial_threshold:
            return "partially_satisfied"
        return "non_existent"

    def _ensure_rationale_pipe(self):
        if self._rationale is None:
            self._rationale = pipeline(
                "text2text-generation",
                model=settings.rationale_model_name,
                max_new_tokens=128,
                num_beams=2,
            )

    def rationale(self, compliance: str, policy: str, label: str) -> str:
        self._ensure_rationale_pipe()
        prompt = (
            "You are auditing a company policy against a compliance clause."
            f"Compliance clause: {compliance}"
            f"Policy clause: {policy}"
            f"Label: {label}."
            "Explain briefly (1 sentence) why this label is appropriate."
        )
        out = self._rationale(prompt)[0]["generated_text"].strip()
        # Keep it to one line without regex
        return " ".join(out.split())

    # NEW: LLM judge that refines PARTIAL cases and returns (label, one-line rationale)
    def _ensure_judge_pipe(self):
        if self._judge is None:
            self._judge = pipeline(
                "text2text-generation",
                model=settings.judge_model_name,
                max_new_tokens=64,
                num_beams=2,
            )

    def judge_label(self, compliance: str, policy: str) -> Tuple[str, str]:
        self._ensure_judge_pipe()
        prompt = (
            "You are a compliance auditor. Decide if the POLICY clause fully satisfies the COMPLIANCE clause."
            "Choose exactly one label: satisfied, partially_satisfied, non_existent."
            "Answer in one line: Label: <label>. Rationale: <short reason>."
            f"Compliance clause: {compliance}"
            f"Policy clause: {policy}"
            "Answer:"
        )
        text = self._judge(prompt)[0]["generated_text"].strip()
        low = text.lower()
        if "non_existent" in low or "non-existent" in low or "non existent" in low:
            label = "non_existent"
        elif "partially_satisfied" in low or "partially satisfied" in low or "partial" in low:
            label = "partially_satisfied"
        elif "satisfied" in low:
            label = "satisfied"
        else:
            label = "partially_satisfied"  # safe default if unclear
        # Extract one-line rationale without regex
        key = "rationale:"
        idx = low.find(key)
        rationale = text[idx + len(key):].strip() if idx != -1 else text
        rationale = " ".join(rationale.split())
        return label, rationale