from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from models import AnalyzeRequest, AnalyzeResponse, ClauseMatch, MatchAlt
from clause_extractor import extract_clauses
from matcher import ClauseMatcher
from settings import settings

app = FastAPI(title="Policy Compliance Analyzer", version="1.0.0")

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
    return {"status": "ok", "embedding_model": settings.embedding_model_name}

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
        best_idx, best_sim = alts[0]
        label = matcher.label_from_similarity(best_sim)
        counts[label] += 1
        match = ClauseMatch(
            id=f"C-{i:04d}",
            text=compliance_clauses[i],
            label=label,
            best_match=MatchAlt(policy_text=policy_clauses[best_idx], similarity=round(best_sim, 3)),
            alternatives=[
                MatchAlt(policy_text=policy_clauses[j], similarity=round(s, 3)) for j, s in alts[1:]
            ],
            rationale=None,
        )
        if req.use_rationale:
            try:
                match.rationale = matcher.rationale(match.text, match.best_match.policy_text, label)
            except Exception:
                match.rationale = None
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
