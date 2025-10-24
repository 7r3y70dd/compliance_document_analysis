from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from settings import settings

class ClauseMatcher:
    def __init__(self):
        # Load embedding model once.
        self.embed = SentenceTransformer(settings.embedding_model_name)
        self._rationale = None

    def _ensure_rationale_pipe(self):
        if self._rationale is None:
            # Small, CPU-friendly model for short justifications.
            self._rationale = pipeline(
                "text2text-generation",
                model=settings.rationale_model_name,
                max_new_tokens=128,
                num_beams=2,
            )

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embed.encode(texts, normalize_embeddings=True))

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
        c_emb = self.encode(compliance_clauses)
        p_emb = self.encode(policy_clauses)
        sims = self.cosine_sim_matrix(c_emb, p_emb)
        results = []
        k = min(top_k, p_emb.shape[0])
        for i in range(sims.shape[0]):
            row = sims[i]
            idx = np.argpartition(-row, range(k))[:k]
            idx = idx[np.argsort(-row[idx])]  # sort by similarity desc
            results.append((i, [(int(j), float(row[j])) for j in idx]))
        return results

    def label_from_similarity(self, sim: float) -> str:
        if sim >= settings.satisfied_threshold:
            return "satisfied"
        if sim >= settings.partial_threshold:
            return "partially_satisfied"
        return "non_existent"

    def rationale(self, compliance: str, policy: str, label: str) -> str:
        self._ensure_rationale_pipe()
        prompt = (
            "You are auditing a company policy against a compliance clause.\n"
            f"Compliance clause: {compliance}\n"
            f"Policy clause: {policy}\n"
            f"Label: {label}.\n"
            "Explain briefly (1-2 sentences) why this label is appropriate."
        )
        out = self._rationale(prompt)[0]["generated_text"].strip()
        return out
