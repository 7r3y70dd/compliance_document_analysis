from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from models import AnalyzeRequest, AnalyzeResponse, ClauseMatch, MatchAlt
from clause_extractor import extract_clauses
from matcher import ClauseMatcher
from settings import settings

app = FastAPI(title="Policy Compliance Analyzer", version="1.1.0")

# CORS for local dev; restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = ClauseMatcher()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_model": settings.embedding_model_name,
        "use_cross_encoder": settings.use_cross_encoder,
        "use_llm_judge": settings.use_llm_judge,
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # 1) Validate sizes early
    if len(req.policy_text) > settings.max_chars or len(req.compliance_text) > settings.max_chars:
        raise HTTPException(status_code=413, detail="Document too large")

    # 2) Extract clauses
    policy_clauses = extract_clauses(req.policy_text)
    compliance_clauses = extract_clauses(req.compliance_text)

    if not policy_clauses or not compliance_clauses:
        raise HTTPException(status_code=400, detail="Could not extract clauses from one or both documents")

    # 3) Match
    pairs = matcher.best_matches(compliance_clauses, policy_clauses, top_k=req.top_k)

    # 4) Build response
    matches = []
    counts = {"satisfied": 0, "partially_satisfied": 0, "non_existent": 0}
    for i, alts in pairs:
        best_idx, best_score = alts[0]

        # --- OLD LABELING (kept for reference) ---
        # label = matcher.label_from_similarity(best_sim)

        # NEW: label from score (sim or CE), then optional judge for partials
        label = matcher.label_from_similarity(best_score)
        rationale = None

        # If enabled, refine partials with LLM judge and capture a one-line rationale
        if settings.use_llm_judge and label == "partially_satisfied":
            try:
                judged_label, judged_rationale = matcher.judge_label(compliance_clauses[i], policy_clauses[best_idx])
                # Overwrite label based on judge; always keep the rationale one-liner
                label = judged_label
                rationale = judged_rationale
            except Exception:
                pass

        # If user requested rationales and none yet, fall back to simple rationale generator
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
            alternatives=[
                MatchAlt(policy_text=policy_clauses[j], similarity=round(s, 3)) for j, s in alts[1:]
            ],
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