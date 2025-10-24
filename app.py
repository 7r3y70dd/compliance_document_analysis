# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import AnalyzeRequest, AnalyzeResponse, ClauseMatch, MatchAlt
from clause_extractor import extract_clauses
from matcher import ClauseMatcher
from settings import settings
from comparator import numeric_downgrade

# NEW: NLI judge
from nli_judge import NLIJudge, stricter  # requires nli_judge.py as provided

app = FastAPI(title="Policy Compliance Analyzer", version="1.3.0")

# CORS for local dev; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = ClauseMatcher()
nli = NLIJudge(settings.nli_model_name) if settings.use_nli_judge else None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model_name,
        "use_cross_encoder": settings.use_cross_encoder,
        "use_llm_judge": settings.use_llm_judge,
        "use_semantic_normalizer": settings.use_semantic_normalizer,
        "use_numeric_check": settings.use_numeric_check,
        "use_nli_judge": settings.use_nli_judge,
        "nli_model": settings.nli_model_name if settings.use_nli_judge else None,
        "nli_annotate_only": settings.nli_annotate_only if settings.use_nli_judge else None,
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # 1) size guard
    if len(req.policy_text) > settings.max_chars or len(req.compliance_text) > settings.max_chars:
        raise HTTPException(status_code=413, detail="Document too large")

    # 2) clause extraction
    policy_clauses = extract_clauses(req.policy_text)
    compliance_clauses = extract_clauses(req.compliance_text)
    if not policy_clauses or not compliance_clauses:
        raise HTTPException(status_code=400, detail="Could not extract clauses from one or both documents")

    # 3) retrieve matches
    pairs = matcher.best_matches(compliance_clauses, policy_clauses, top_k=req.top_k)

    # 4) scoring & labeling
    matches = []
    counts = {"satisfied": 0, "partially_satisfied": 0, "non_existent": 0}

    for i, alts in pairs:
        best_idx, best_score = alts[0]

        # initial label from similarity (bi-encoder or cross-encoder score)
        label = matcher.label_from_similarity(best_score)
        rationale = None

        # numeric downgrade first (e.g., 24h vs 48h)
        if settings.use_numeric_check and label == "satisfied":
            try:
                downgrade, reason = numeric_downgrade(compliance_clauses[i], policy_clauses[best_idx])
                if downgrade:
                    label = "partially_satisfied"
                    rationale = reason
            except Exception:
                pass

        # NLI judge (directional check: does policy entail compliance?)
        if nli is not None:
            try:
                nli_label, nli_scores = nli.classify(policy_clauses[best_idx], compliance_clauses[i])
                nli_note = (
                    f"NLI verdict={nli_label} "
                    f"(ent={nli_scores['entailment']:.2f}, neu={nli_scores['neutral']:.2f}, con={nli_scores['contradiction']:.2f})"
                )
                rationale = (rationale + " | " if rationale else "") + nli_note

                # If not annotate-only, constrain label toward stricter outcome (never upgrades)
                if not settings.nli_annotate_only:
                    label = stricter(label, nli_label)
            except Exception:
                # keep going even if NLI fails
                pass

        # optional simple rationale (legacy) if requested and none present
        if req.use_rationale and rationale is None:
            try:
                rationale = matcher.rationale(compliance_clauses[i], policy_clauses[best_idx], label)
            except Exception:
                rationale = None

        counts[label] += 1

        match = ClauseMatch(
            id=f"C-{i:04d}",
            text=compliance_clauses[i],
            label=label,
            best_match=MatchAlt(policy_text=policy_clauses[best_idx], similarity=round(best_score, 3)),
            alternatives=[MatchAlt(policy_text=policy_clauses[j], similarity=round(s, 3)) for j, s in alts[1:]],
            rationale=rationale,
        )
        matches.append(match)

    return AnalyzeResponse(overall=counts, clauses=matches)

@app.post("/analyze-multipart", response_model=AnalyzeResponse)
async def analyze_multipart(
    policy: UploadFile = File(...),
    compliance: UploadFile = File(...),
    top_k: int = Form(3),
    use_rationale: bool = Form(False),
):
    policy_text = (await policy.read()).decode("utf-8", errors="ignore")
    compliance_text = (await compliance.read()).decode("utf-8", errors="ignore")
    req = AnalyzeRequest(
        policy_text=policy_text,
        compliance_text=compliance_text,
        top_k=top_k,
        use_rationale=use_rationale,
    )
    return analyze(req)
